/*
* Scene.h
*
* Copyright (c) 2014-2015 SEACAVE
*
* Author(s):
*
*      cDc <cdc.seacave@gmail.com>
*
*
* This program is free software: you can redistribute it and/or modify
* it under the terms of the GNU Affero General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU Affero General Public License for more details.
*
* You should have received a copy of the GNU Affero General Public License
* along with this program.  If not, see <http://www.gnu.org/licenses/>.
*
*
* Additional Terms:
*
*      You are required to preserve legal notices and author attributions in
*      that material or in the Appropriate Legal Notices displayed by works
*      containing it.
*/

#ifndef _MVS_SCENE_H_
#define _MVS_SCENE_H_


// I N C L U D E S /////////////////////////////////////////////////

#include "DepthMap.h"

// D E F I N E S ///////////////////////////////////////////////////


// S T R U C T S ///////////////////////////////////////////////////

namespace MVS {

class MVS_API Scene
{
public:
	PlatformArr platforms; // camera platforms, each containing the mounted cameras and all known poses
	ImageArr images; // images, each referencing a platform's camera pose
	PointCloud pointcloud; // point-cloud (sparse or dense), each containing the point position and the views seeing it

	unsigned nCalibratedImages; // number of valid images

	unsigned nMaxThreads; // maximum number of threads used to distribute the work load

public:
	inline Scene(unsigned _nMaxThreads=0) : nMaxThreads(Thread::getMaxThreads(_nMaxThreads)) {}

	void Release();
	bool IsEmpty() const;

	bool LoadInterface(const String& fileName);
	bool SaveInterface(const String& fileName) const;

	bool Load(const String& fileName);
	bool Save(const String& fileName, ARCHIVE_TYPE type=ARCHIVE_BINARY_ZIP) const;

	bool SelectNeighborViews(uint32_t ID, IndexArr& points, unsigned nMinViews=3, unsigned nMinPointViews=2, float fOptimAngle=FD2R(12));
	static bool FilterNeighborViews(ViewScoreArr& neighbors, float fMinArea=0.12f, float fMinScale=0.2f, float fMaxScale=2.4f, float fMinAngle=FD2R(3), float fMaxAngle=FD2R(45), unsigned nMaxViews=12);

	bool ExportCamerasMLP(const String& fileName, const String& fileNameScene) const;

	// Dense reconstruction
	bool DenseReconstruction(bool USE_CUDA=false);
	void DenseReconstructionEstimate(void*,bool USE_CUDA=false);
	void DenseReconstructionFilter(void*);

	#ifdef _USE_BOOST
	// implement BOOST serialization
	template <class Archive>
	void serialize(Archive& ar, const unsigned int /*version*/) {
		ar & platforms;
		ar & images;
		ar & pointcloud;
	}
	#endif
};
/*----------------------------------------------------------------*/

} // namespace MVS

#endif // _MVS_SCENE_H_
