#include "cuda_runtime.h"  
#include "device_launch_parameters.h" 

struct MVS_API DepthEstimator_CUDA 
{
	enum { TexelChannels = 1 };
	enum { nSizeHalfWindow = 3 };
	enum { nSizeWindow = nSizeHalfWindow*2+1 };
	enum { nTexels = nSizeWindow*nSizeWindow*TexelChannels };


	typedef Eigen::Matrix<float,nTexels,1> TexelVec;
	struct NeighborData 
	{
		Depth depth;
		Normal normal;
		inline NeighborData() {}
		inline NeighborData(Depth d, const Normal& n) : depth(d), normal(n) {}
	};

	struct ViewData 
    {
		const DepthData::ViewData& view;
		const Matrix3x3 Hl;   //
		const Vec3 Hm;	      // constants during per-pixel loops
		const Matrix3x3 Hr;   //
		inline ViewData() : view(*((const DepthData::ViewData*)this)) {}
		inline ViewData(const DepthData::ViewData& image0, const DepthData::ViewData& image1)
			: view(image1),
			Hl(image1.camera.K * image1.camera.R * image0.camera.R.t()),
			Hm(image1.camera.K * image1.camera.R * (image0.camera.C - image1.camera.C)),
			Hr(image0.camera.K.inv()) {}
	};

	CLISTDEF0(NeighborData) neighborsData; // neighbor pixel depths to be used for smoothing
	CLISTDEF0(ImageRef) neighbors; // neighbor pixels coordinates to be processed
	volatile Thread::safe_t& idxPixel; // current image index to be processed
	Vec3 X0;	      //
	ImageRef lt0;	  // constants during one pixel loop
	float normSq0;	  //
	TexelVec texels0; //
	TexelVec texels1;
	FloatArr scores;
	DepthMap& depthMap0;
	NormalMap& normalMap0;
	ConfidenceMap& confMap0;

	const CLISTDEF0(ViewData) images; // neighbor images used
	const DepthData::ViewData& image0;
	const Image32F& image0Sum; // integral image used to fast compute patch mean intensity
	const MapRefArr& coords;
	const Image8U::Size size;
	const IDX idxScore;
	const ENDIRECTION dir;
	const Depth dMin, dMax;

	DepthEstimator(DepthData& _depthData0, volatile Thread::safe_t& _idx, const Image32F& _image0Sum, const MapRefArr& _coords, ENDIRECTION _dir);

	bool PreparePixelPatch(const ImageRef&);
	bool FillPixelPatch(const ImageRef&);
	float ScorePixel(Depth, const Normal&);
	void ProcessPixel(IDX idx);
	
	inline float GetImage0Sum(const ImageRef& p0) 
    {
		const ImageRef p1(p0.x+nSizeWindow, p0.y);
		const ImageRef p2(p0.x, p0.y+nSizeWindow);
		const ImageRef p3(p0.x+nSizeWindow, p0.y+nSizeWindow);
		return image0Sum(p3) - image0Sum(p2) - image0Sum(p1) + image0Sum(p0);
	}

	inline Matrix3x3f ComputeHomographyMatrix(const ViewData& img, Depth depth, const Normal& normal) const {
		#if 0
		// compute homography matrix
		const Matrix3x3f H(img.view.camera.K*HomographyMatrixComposition(image0.camera, img.view.camera, Vec3(normal), Vec3(X0*depth))*image0.camera.K.inv());
		#else
		// compute homography matrix as above, caching some constants
		const Vec3 n(normal);
		return (img.Hl + img.Hm * (n.t()*INVERT(n.dot(X0)*depth))) * img.Hr;
		#endif
	}

	static inline CLISTDEF0(ViewData) InitImages(const DepthData& depthData) 
    {
		CLISTDEF0(ViewData) images(0, depthData.images.GetSize()-1);
		const DepthData::ViewData& image0(depthData.images.First());
		for (IDX i=1; i<depthData.images.GetSize(); ++i)
			images.AddConstruct(image0, depthData.images[i]);
		return images;
	}

	static inline Point3 ComputeRelativeC(const DepthData& depthData) 
    {
		return depthData.images[1].camera.R*(depthData.images[0].camera.C-depthData.images[1].camera.C);
	}
	static inline Matrix3x3 ComputeRelativeR(const DepthData& depthData) 
    {
		RMatrix R;
		ComputeRelativeRotation(depthData.images[0].camera.R, depthData.images[1].camera.R, R);
		return R;
	}

	// generate random depth and normal
	static inline Depth RandomDepth(Depth dMin, Depth dMax) 
	{
		ASSERT(dMin > 0);
		return randomRange(dMin, dMax);
	}
	static inline Normal RandomNormal() 
	{
		const float a1Min = FD2R(0.f);
		const float a1Max = FD2R(360.f);
		const float a2Min = FD2R(120.f);
		const float a2Max = FD2R(180.f);
		Normal normal;
		Dir2Normal(Point2f(randomRange(a1Min,a1Max), randomRange(a2Min,a2Max)), normal);
		ASSERT(normal.z < 0);
		return normal;
	}

	// encode/decode NCC score and refinement level in one float
	static inline float EncodeScoreScale(float score, unsigned invScaleRange=0) 
    {
		ASSERT(score >= 0.f && score <= 2.01f);
		return score*0.1f+(float)invScaleRange;
	}
	static inline unsigned DecodeScale(float score) 
    {
		return (unsigned)FLOOR2INT(score);
	}
	static inline unsigned DecodeScoreScale(float& score) 
    {
		const unsigned invScaleRange(DecodeScale(score));
		score = (score-(float)invScaleRange)*10.f;
		//ASSERT(score >= 0.f && score <= 2.01f); //problems in multi-threading
		return invScaleRange;
	}
	static inline float DecodeScore(float score) 
    {
		DecodeScoreScale(score);
		return score;
	}

	// Encodes/decodes a normalized 3D vector in two parameters for the direction
	template<typename T, typename TR>
	static inline void Normal2Dir(const TPoint3<T>& d, TPoint2<TR>& p) 
    {
		// empirically tested
		ASSERT(ISEQUAL(norm(d), T(1)));
		p.y = TR(atan2(sqrt(d.x*d.x + d.y*d.y), d.z));
		p.x = TR(atan2(d.y, d.x));
	}
	template<typename T, typename TR>
	static inline void Dir2Normal(const TPoint2<T>& p, TPoint3<TR>& d) 
    {
		// empirically tested
		const T siny(sin(p.y));
		d.x = TR(cos(p.x)*siny);
		d.y = TR(sin(p.x)*siny);
		d.z = TR(cos(p.y));
		ASSERT(ISEQUAL(norm(d), TR(1)));
	}

	static void MapMatrix2ZigzagIdx(const Image8U::Size& size, DepthEstimator::MapRefArr& coords, BitMatrix& mask, int rawStride=16);

	const float smoothBonusDepth, smoothBonusNormal;
	const float smoothSigmaDepth, smoothSigmaNormal;
	const float thMagnitudeSq;
	const float angle1Range, angle2Range;
	const float thConfSmall, thConfBig, thConfIgnore;
	static const float scaleRanges[12];
};