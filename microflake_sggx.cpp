#include <mitsuba/core/frame.h>
#include <mitsuba/render/phase.h>
#include <mitsuba/render/medium.h>
#include <mitsuba/render/sampler.h>

/// Generate a few statistics related to the implementation?
// #define MICROFLAKE_STATISTICS 1

MTS_NAMESPACE_BEGIN

#if defined(MICROFLAKE_STATISTICS)
static StatsCounter avgSampleIterations("Micro-flake SGGX model",
		"Average rejection sampling iterations", EAverage);
#endif

namespace
{
    void calcFiberLikeSGGXMatrix(Float& Sxx, Float& Syy, Float& Szz,
                        Float& Sxy, Float& Sxz, Float& Syz,
                        Float roughness,
                        const Vector& omega3)
    {
        Float roughness2 = roughness*roughness;
        Sxx = roughness2*omega3.x*omega3.x + omega3.y*omega3.y + omega3.z*omega3.z;
		Sxy = roughness2*omega3.x*omega3.y - omega3.x*omega3.y;
		Sxz = roughness2*omega3.x*omega3.z - omega3.x*omega3.z;
		Syy = roughness2*omega3.y*omega3.y + omega3.x*omega3.x + omega3.z*omega3.z;
		Syz = roughness2*omega3.y*omega3.z - omega3.y*omega3.z;
		Szz = roughness2*omega3.z*omega3.z + omega3.x*omega3.x + omega3.y*omega3.y;
    }

    Float sigma(const Vector& wi,
                Float S_xx, Float S_yy, Float S_zz,
                Float S_xy, Float S_xz, Float S_yz)
    {
        const Float sigma_squared = wi.x*wi.x*S_xx + wi.y*wi.y*S_yy + wi.z*wi.z*S_zz
            + 2.0f * (wi.x*wi.y*S_xy + wi.x*wi.z*S_xz + wi.y*wi.z*S_yz);
        return (sigma_squared > 0.0f) ? sqrtf(sigma_squared) : 0.0f; // conditional to avoid numerical errors
    }

    Float D(const Vector& wm,
            Float S_xx, Float S_yy, Float S_zz,
            Float S_xy, Float S_xz, Float S_yz)
    {
        const Float detS = S_xx*S_yy*S_zz - S_xx*S_yz*S_yz - S_yy*S_xz*S_xz - S_zz*S_xy*S_xy + 2.0f*S_xy*S_xz*S_yz;
        const Float den = wm.x*wm.x*(S_yy*S_zz-S_yz*S_yz) + wm.y*wm.y*(S_xx*S_zz-S_xz*S_xz) + wm.z*wm.z*(S_xx*S_yy-S_xy*S_xy)
            + 2.0f*(wm.x*wm.y*(S_xz*S_yz-S_zz*S_xy) + wm.x*wm.z*(S_xy*S_yz-S_yy*S_xz) + wm.y*wm.z*(S_xy*S_xz-S_xx*S_yz));
        const Float D = powf(fabsf(detS), 1.5f) / (M_PI*den*den);
        return D;
    }

    void buildOrthonormalBasis(Vector& omega_1, Vector& omega_2, const Vector& omega_3)
    {
        if(omega_3.z < -0.9999999f)
        {
            omega_1 = Vector(0.0f, -1.0f, 0.0f);
            omega_2 = Vector(-1.0f, 0.0f, 0.0f);
        } else {
            const Float a = 1.0f /(1.0f + omega_3.z);
            const Float b = -omega_3.x*omega_3.y*a;
            omega_1 = Vector(1.0f - omega_3.x*omega_3.x*a, b, -omega_3.x);
            omega_2 = Vector(b, 1.0f - omega_3.y*omega_3.y*a, -omega_3.y);
        }
    }

    Vector sample_VNDF(const Vector& wi,
                       Float S_xx, Float S_yy, Float S_zz,
                       Float S_xy, Float S_xz, Float S_yz,
                       Float U1, Float U2)
    {
        // generate sample (u, v, w)
        const Float r = sqrtf(U1);
        const Float phi = 2.0f*M_PI*U2;
        const Float u = r*cosf(phi);
        const Float v= r*sinf(phi);
        const Float w = sqrtf(1.0f - u*u - v*v);

        // build orthonormal basis
        Vector wk, wj;
        buildOrthonormalBasis(wk, wj, wi);
        // project S in this basis
        const Float S_kk = wk.x*wk.x*S_xx + wk.y*wk.y*S_yy + wk.z*wk.z*S_zz
        + 2.0f * (wk.x*wk.y*S_xy + wk.x*wk.z*S_xz + wk.y*wk.z*S_yz);
        const Float S_jj = wj.x*wj.x*S_xx + wj.y*wj.y*S_yy + wj.z*wj.z*S_zz
        + 2.0f * (wj.x*wj.y*S_xy + wj.x*wj.z*S_xz + wj.y*wj.z*S_yz);
        const Float S_ii = wi.x*wi.x*S_xx + wi.y*wi.y*S_yy + wi.z*wi.z*S_zz
        + 2.0f * (wi.x*wi.y*S_xy + wi.x*wi.z*S_xz + wi.y*wi.z*S_yz);
        const Float S_kj = wk.x*wj.x*S_xx + wk.y*wj.y*S_yy + wk.z*wj.z*S_zz
        + (wk.x*wj.y + wk.y*wj.x)*S_xy
        + (wk.x*wj.z + wk.z*wj.x)*S_xz
        + (wk.y*wj.z + wk.z*wj.y)*S_yz;
        const Float S_ki = wk.x*wi.x*S_xx + wk.y*wi.y*S_yy + wk.z*wi.z*S_zz
        + (wk.x*wi.y + wk.y*wi.x)*S_xy + (wk.x*wi.z + wk.z*wi.x)*S_xz + (wk.y*wi.z + wk.z*wi.y)*S_yz;
        const Float S_ji = wj.x*wi.x*S_xx + wj.y*wi.y*S_yy + wj.z*wi.z*S_zz
        + (wj.x*wi.y + wj.y*wi.x)*S_xy
        + (wj.x*wi.z + wj.z*wi.x)*S_xz
        + (wj.y*wi.z + wj.z*wi.y)*S_yz;
        // compute normal
        Float sqrtDetSkji = sqrtf(fabsf(S_kk*S_jj*S_ii - S_kj*S_kj*S_ii - S_ki*S_ki*S_jj - S_ji*S_ji*S_kk + 2.0f*S_kj*S_ki*S_ji));
        Float inv_sqrtS_ii = 1.0f / sqrtf(S_ii);
        Float tmp = sqrtf(S_jj*S_ii-S_ji*S_ji);
        Vector Mk(sqrtDetSkji/tmp, 0.0f, 0.0f);
        Vector Mj(-inv_sqrtS_ii*(S_ki*S_ji-S_kj*S_ii)/tmp, inv_sqrtS_ii*tmp, 0);
        Vector Mi(inv_sqrtS_ii*S_ki, inv_sqrtS_ii*S_ji, inv_sqrtS_ii*S_ii);
        Vector wm_kji = normalize(u*Mk+v*Mj+w*Mi);
        // rotate back to world basis
        return wm_kji.x * wk + wm_kji.y * wj + wm_kji.z * wi;
    }

    Float eval_specular(const Vector& wi, const Vector& wo,
                        Float S_xx, Float S_yy, Float S_zz,
                        Float S_xy, Float S_xz, Float S_yz)
    {
        Vector wh = normalize(wi + wo);
        return 0.25f * D(wh, S_xx, S_yy, S_zz, S_xy, S_xz, S_yz) / sigma(wi, S_xx, S_yy, S_zz, S_xy, S_xz, S_yz);
    }

    Vector sample_specular(const Vector& wi,
                           Float S_xx, Float S_yy, Float S_zz,
                           Float S_xy, Float S_xz, Float S_yz,
                           Float U1, Float U2)
    {
        // sample VNDF
        const Vector wm = sample_VNDF(wi, S_xx, S_yy, S_zz, S_xy, S_xz, S_yz, U1, U2);
        // specular reflection
        const Vector wo = -wi + 2.0f * wm * dot(wm, wi);
        return wo;
    }

    Float eval_diffuse(const Vector& wi, const Vector& wo,
        Float S_xx, Float S_yy, Float S_zz,
        Float S_xy, Float S_xz, Float S_yz,
        Float U1, Float U2)
    {
        // sample VNDF
        const Vector wm = sample_VNDF(wi, S_xx, S_yy, S_zz, S_xy, S_xz, S_yz, U1, U2);
        // eval diffuse
        return 1.0f / M_PI * fmaxf(0.0f, dot(wo, wm));
    }

    Vector sample_diffuse(const Vector& wi,
        Float S_xx, Float S_yy, Float S_zz,
        Float S_xy, Float S_xz, Float S_yz,
        Float U1, Float U2, Float U3, Float U4)
    {
        // sample VNDF
        const Vector wm = sample_VNDF(wi, S_xx, S_yy, S_zz, S_xy, S_xz, S_yz, U1, U2);
        // sample diffuse reflection
        Vector w1, w2;
        buildOrthonormalBasis(w1, w2, wm);
        Float r1 = 2.0f*U3 - 1.0f;
        Float r2 = 2.0f*U4 - 1.0f;
        // concentric map code from
        // http://psgraphics.blogspot.ch/2011/01/improved-code-for-concentric-map.html
        Float phi, r;
        if(r1 == 0 && r2 == 0) {
            r = phi = 0;
        } else if(r1*r1 > r2*r2) {
            r = r1;
            phi = (M_PI/4.0f) * (r2/r1);
        } else {
            r = r2;
            phi = (M_PI/2.0f) - (r1/r2) * (M_PI/4.0f);
        }
        Float x = r*cosf(phi);
        Float y = r*sinf(phi);
        Float z = sqrtf(1.0f - x*x - y*y);
        Vector wo = x*w1 + y*w2 + z*wm;
        return wo;
    }

}

/*!\plugin{microflake}{Micro-flake phase function}
 * \parameters{
 *     \parameter{roughness}{\Float}{
 *       The roughness of the fibers in the medium.
 *     }
 * }
 *
 *
 * This plugin implements the SGGX microflake phase function described in
 * ``The SGGX microflake distribution'' by Eric Heitz, Jonathan Dupuy,
 * Cyril Crassin, and Carsten Dachsbacher
 * \cite{Heitz2015}.

 */
class MicroflakePhaseFunction : public PhaseFunction {
public:
	MicroflakePhaseFunction(const Properties &props) : PhaseFunction(props) {
		/// Standard deviation of the flake distribution
        m_roughness = props.getFloat("roughness", 1.0f);
	}

	MicroflakePhaseFunction(Stream *stream, InstanceManager *manager)
		: PhaseFunction(stream, manager) {
        m_roughness = stream->readFloat();
		configure();
	}

	virtual ~MicroflakePhaseFunction() { }


	void configure() {
		PhaseFunction::configure();
		m_type = EAnisotropic | ENonSymmetric;
	}

	void serialize(Stream *stream, InstanceManager *manager) const {
		PhaseFunction::serialize(stream, manager);
        stream->writeFloat(m_roughness);
	}

	Float eval(const PhaseFunctionSamplingRecord &pRec) const {
		if (pRec.mRec.orientation.isZero()) {
			/* What to do when the local orientation is undefined */
			#if 0
				return 1.0f / (4 * M_PI);
			#else
				return 0.0f;
			#endif
		}

        Frame frame(pRec.mRec.orientation);

        Vector omega3 = frame.t;
        Float Sxx, Sxy, Sxz, Syy, Syz, Szz;
        calcFiberLikeSGGXMatrix(Sxx, Syy, Szz, Sxy, Sxz, Syz, m_roughness, omega3);

		Vector wi = frame.toLocal(pRec.wi);
		Vector wo = frame.toLocal(pRec.wo);
        Float ret = eval_specular(wi, wo, Sxx, Syy, Szz, Sxy, Sxz, Syz);

		//Vector H = wi + wo;
		//Float length = H.length();

		//if (length == 0)
		//	return 0.0f;

		return ret;
	}

	inline Float sample(PhaseFunctionSamplingRecord &pRec, Sampler *sampler) const {
		if (pRec.mRec.orientation.isZero()) {
			/* What to do when the local orientation is undefined */
			#if 0
				pRec.wo = warp::squareToUniformSphere(sampler->next2D());
				return 1.0f;
			#else
				return 0.0f;
			#endif
		}
        Frame frame(pRec.mRec.orientation);
        Vector omega3 = frame.t;
        Float Sxx, Sxy, Sxz, Syy, Syz, Szz;
        calcFiberLikeSGGXMatrix(Sxx, Syy, Szz, Sxy, Sxz, Syz, m_roughness, omega3);

		Vector wi = frame.toLocal(pRec.wi);

		#if defined(MICROFLAKE_STATISTICS)
			avgSampleIterations.incrementBase();
		#endif

        Point2 point = sampler->next2D();
		pRec.wo = normalize(sample_specular(wi, Sxx, Syy, Szz, Sxy, Sxz, Syz, point.x, point.y));

		return 1.0f;
	}

	Float sample(PhaseFunctionSamplingRecord &pRec,
			Float &pdf, Sampler *sampler) const {
		if (sample(pRec, sampler) == 0) {
			pdf = 0; return 0.0f;
		}
		pdf = eval(pRec);
		return 1.0f;
	}

	bool needsDirectionallyVaryingCoefficients() const { return true; }

	Float sigmaDir(Float cosTheta) const {
		// Scaled such that replacing an isotropic phase function with an
		// isotropic microflake distribution does not cause changes
        Vector omega3 = Vector(0.0f, 1.0f, 0.0f);
        Float Sxx, Sxy, Sxz, Syy, Syz, Szz;
        calcFiberLikeSGGXMatrix(Sxx, Syy, Szz, Sxy, Sxz, Syz, m_roughness, omega3);

        Vector wi = Vector(sqrtf(1 - (cosTheta*cosTheta)), 0, cosTheta);
		return  sigma(wi,Sxx,Syy,Szz,Sxy,Sxz,Syz);
	}

	Float sigmaDirMax() const {
		return sigmaDir(0);
	}

	std::string toString() const {
		std::ostringstream oss;
		oss << "MicroflakeSGGXPhaseFunction[" << endl
			<< "   roughness = " << m_roughness << endl
			<< "]";
		return oss.str();
	}

	MTS_DECLARE_CLASS()
private:
    Float m_roughness;
};

MTS_IMPLEMENT_CLASS_S(MicroflakePhaseFunction, false, PhaseFunction)
MTS_EXPORT_PLUGIN(MicroflakePhaseFunction, "Microflake SGGX phase function");
MTS_NAMESPACE_END
