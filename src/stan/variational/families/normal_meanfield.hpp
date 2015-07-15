#ifndef STAN_VARIATIONAL_NORMAL_MEANFIELD_HPP
#define STAN_VARIATIONAL_NORMAL_MEANFIELD_HPP

#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/math/prim/scal/fun/constants.hpp>

#include <stan/math/prim/mat/meta/get.hpp>
#include <stan/math/prim/arr/meta/get.hpp>
#include <stan/math/prim/mat/meta/length.hpp>
#include <stan/math/prim/mat/meta/is_vector.hpp>
#include <stan/math/prim/mat/meta/is_vector_like.hpp>
#include <stan/math/prim/mat/fun/value_of_rec.hpp>
#include <stan/math/prim/scal/prob/normal_rng.hpp>

#include <stan/math/prim/scal/err/check_size_match.hpp>
#include <stan/math/prim/scal/err/check_not_nan.hpp>

#include <stan/model/util.hpp>

#include <stan/variational/base_family.hpp>
#include <algorithm>
#include <ostream>
#include <vector>

namespace stan {

  namespace variational {

    class normal_meanfield : public base_family {
    private:
      Eigen::VectorXd mu_;
      Eigen::VectorXd omega_; // log standard deviation vector
      int dimension_;

    public:
      // Constructors
      explicit normal_meanfield(size_t dimension) :
        dimension_(dimension) {
        mu_     = Eigen::VectorXd::Zero(dimension_);
        // initializing omega = 0 means sigma = 1
        omega_  = Eigen::VectorXd::Zero(dimension_);
      }

      // TODO do I need all these other constructors?
      explicit normal_meanfield(const Eigen::VectorXd& cont_params) :
        dimension_(cont_params.size()) {
        set_mu(cont_params);
        // initializing omega = 0 means sigma = 1
        omega_  = Eigen::VectorXd::Zero(dimension_);
      }

      normal_meanfield(const Eigen::VectorXd& mu,
                       const Eigen::VectorXd& omega) :
        dimension_(mu.size()) {
        set_mu(mu);
        set_omega(omega);
      }

      int dimension() const { return dimension_; }
      const Eigen::VectorXd& mu() const { return mu_; }
      const Eigen::VectorXd& omega() const { return omega_; }

      void set_mu(const Eigen::VectorXd& mu) {
        static const char* function =
          "stan::variational::normal_meanfield::set_mu";

        stan::math::check_size_match(function,
                               "Dimension of input vector", mu.size(),
                               "Dimension of current vector", dimension_);
        stan::math::check_not_nan(function, "Input vector", mu);
        mu_ = mu;
      }

      void set_omega(const Eigen::VectorXd& omega) {
        static const char* function =
          "stan::variational::normal_meanfield::set_omega";

        stan::math::check_size_match(function,
                               "Dimension of input vector", omega.size(),
                               "Dimension of current vector", dimension_);
        stan::math::check_not_nan(function, "Input vector", omega);
        omega_ = omega;
      }

      normal_meanfield square() const {
        return normal_meanfield(Eigen::VectorXd(mu_.array().square()),
                                Eigen::VectorXd(omega_.array().square()));
      }

      normal_meanfield sqrt() const {
        return normal_meanfield(Eigen::VectorXd(mu_.array().sqrt()),
                                Eigen::VectorXd(omega_.array().sqrt()));
      }

      normal_meanfield operator=(const normal_meanfield& rhs) {
        static const char* function =
          "stan::variational::normal_meanfield::operator=";

        stan::math::check_size_match(function,
                             "Dimension of lhs", dimension_,
                             "Dimension of rhs", rhs.dimension());

        mu_ = rhs.mu();
        omega_ = rhs.omega();
        return *this;
      }

      normal_meanfield operator+=(const normal_meanfield& rhs) {
        static const char* function =
          "stan::variational::normal_meanfield::operator+=";

        stan::math::check_size_match(function,
                             "Dimension of lhs", dimension_,
                             "Dimension of rhs", rhs.dimension());

        mu_ += rhs.mu();
        omega_ += rhs.omega();
        return *this;
      }

      normal_meanfield operator/=(const normal_meanfield& rhs) {
        static const char* function =
          "stan::variational::normal_meanfield::operator/=";

        stan::math::check_size_match(function,
                             "Dimension of lhs", dimension_,
                             "Dimension of rhs", rhs.dimension());

        mu_.array() /= rhs.mu().array();
        omega_.array() /= rhs.omega().array();
        return *this;
      }

      normal_meanfield operator+=(double x) {
        mu_.array() += x;
        omega_.array() += x;
        return *this;
      }

      normal_meanfield operator*=(double x) {
        mu_ *= x;
        omega_ *= x;
        return *this;
      }

      const Eigen::VectorXd& mean() const {
        return mu();
      }

      // 0.5 * dim * (1+log2pi) + 0.5 * log det diag(sigma^2) =
      // 0.5 * dim * (1+log2pi) + sum(log(sigma)) =
      // 0.5 * dim * (1+log2pi) + sum(omega)
      double entropy() const {
        return 0.5 * dimension_ * (1.0 + stan::math::LOG_TWO_PI) + omega_.sum();
      }

      // Implements S^{-1}(eta) = sigma * eta + \mu
      Eigen::VectorXd transform(const Eigen::VectorXd& eta) const {
        static const char* function =
          "stan::variational::normal_meanfield::transform";

        stan::math::check_size_match(function,
                         "Dimension of mean vector", dimension_,
                         "Dimension of input vector", eta.size() );
        stan::math::check_not_nan(function, "Input vector", eta);

        // exp(omega) * eta + mu
        return eta.array().cwiseProduct(omega_.array().exp()) + mu_.array();
      }

      template <class BaseRNG>
      Eigen::VectorXd sample(BaseRNG& rng) const {
        Eigen::VectorXd eta(dimension_);

        for (int d = 0; d < dimension_; ++d) {
          eta(d) = stan::math::normal_rng(0, 1, rng);
        }

        return transform(eta);
      }

      /**
       * Calculates the "blackbox" gradient with respect to the location vector
       * (mu) and the log-std vector (omega).
       *
       * @tparam M                     class of model
       * @tparam BaseRNG               class of random number generator
       * @param  elbo_grad             parameters to store "blackbox" gradient
       * @param  cont_params           continuous parameters
       * @param  n_monte_carlo_grad    number of samples for gradient computation
       * @param  print_stream          stream for convergence assessment output
       */
      template <class M, class BaseRNG>
      void calc_grad(normal_meanfield& elbo_grad,
                     M& m,
                     Eigen::VectorXd& cont_params,
                     int n_monte_carlo_grad,
                     BaseRNG& rng,
                     std::ostream* print_stream) const {
        static const char* function =
          "stan::variational::normal_meanfield::calc_grad";

        stan::math::check_size_match(function,
                        "Dimension of elbo_grad", elbo_grad.dimension(),
                        "Dimension of variational q", dimension_);
        stan::math::check_size_match(function,
                        "Dimension of variational q", dimension_,
                        "Dimension of variables in model", cont_params.size());

        // Initialize everything to zero
        Eigen::VectorXd mu_grad    = Eigen::VectorXd::Zero(dimension_);
        Eigen::VectorXd omega_grad = Eigen::VectorXd::Zero(dimension_);
        double tmp_lp = 0.0;
        Eigen::VectorXd tmp_mu_grad(dimension_);
        Eigen::VectorXd eta(dimension_);
        Eigen::VectorXd zeta(dimension_);

        // Naive Monte Carlo integration
        for (int i = 0; i < n_monte_carlo_grad; ++i) {
          // Draw from standard normal and transform to real-coordinate space
          for (int d = 0; d < dimension_; ++d) {
            eta(d) = stan::math::normal_rng(0, 1, rng);
          }
          zeta = transform(eta);

          // Compute gradient step in real-coordinate space
          stan::model::gradient(m, zeta, tmp_lp, tmp_mu_grad, print_stream);
          stan::math::check_not_nan(function, "tmp_lp", tmp_lp);
          stan::math::check_finite(function, "tmp_lp", tmp_lp);
          stan::math::check_not_nan(function, "tmp_mu_grad", tmp_mu_grad);
          stan::math::check_finite(function, "tmp_mu_grad", tmp_mu_grad);

          // Update gradient parameters
          mu_grad += tmp_mu_grad;
          omega_grad.array() += tmp_mu_grad.array().cwiseProduct(eta.array());
        }
        mu_grad    /= static_cast<double>(n_monte_carlo_grad);
        omega_grad /= static_cast<double>(n_monte_carlo_grad);

        // Multiply by exp(omega)
        omega_grad.array() =
          omega_grad.array().cwiseProduct(omega_.array().exp());

        // Add gradient of entropy term (just equal to element-wise 1 here)
        omega_grad.array() += 1.0;

        // Set parameters to argument
        elbo_grad.set_mu(mu_grad);
        elbo_grad.set_omega(omega_grad);
      }
    };

    normal_meanfield operator+(normal_meanfield lhs, const normal_meanfield& rhs) { return lhs += rhs; }
    normal_meanfield operator/(normal_meanfield lhs, const normal_meanfield& rhs) { return lhs /= rhs; }
    normal_meanfield operator+(double x, normal_meanfield rhs) { return rhs += x; }
    normal_meanfield operator*(double x, normal_meanfield rhs) { return rhs *= x; }
  }  // variational
}  // stan

#endif
