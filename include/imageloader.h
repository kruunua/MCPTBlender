/**
 * @file imageloader.h
 * @author E. Denisova
 * @date 29/2/2024
 * @version 1.0
**/

#ifndef IMAGELOADER_H
#define IMAGELOADER_H

#include <vector>
#include <string>
#include "curvepredictor.h"

class ImageLoader
{
public:
    static std::vector<float> loadImage(const std::string &fileName,
                                        int &w, int &h);
    static std::vector<CurveParam> loadCurves(const std::string &name0,
                                              const std::string &name1,
                                              int &w, int &h);
    static std::vector<int> loadWeights(const std::string &name, int w, int h);
    static bool saveExr(const std::vector<float> &data, int w, int h,
                        const std::string &name);
    static bool saveExr(const std::vector<CurveParam> &data, int w, int h,
                        const std::string &name0, const std::string &name1);
    static bool saveExr(const std::vector<int> &data, int w, int h,
                        const std::string &name);
    static void gaussianBlur(const std::vector<float> &src,
                             std::vector<float> &dst,
                             int w, int h, int kernelSize,
                             const std::vector<float> &var);
    static float mse(const std::vector<float> &img1,
                     const std::vector<float> &img2);
    static float avg(const std::vector<float> &img);
    static std::vector<float> mseVector(const std::vector<float> &img1,
                                        const std::vector<float> &img2);
    static std::vector<float> diff(const std::vector<float> &img,
                                   const std::vector<float> &ref);
};

#endif // IMAGELOADER_H
