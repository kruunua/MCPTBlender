/**
 * @file curvepredictor.h
 * @author E. Denisova
 * @date 29/2/2024
 * @version 1.0
**/

#ifndef CURVEPREDICTOR_H
#define CURVEPREDICTOR_H

#include <vector>
#include <utility>

typedef std::pair<float, float> CurveParam;

class CurvePredictor
{
public:
    static std::vector<float> sure(const std::vector<float> &denoised,
                                   const std::vector<float> &noisy,
                                   int w, int h,
                                   const std::vector<float> &var,
                                   bool useOptiX, bool hdr, bool cleanAux);
    static int denoisedWeight(float sure, const CurveParam &outs,
                              int minWeight);
    static void blend(const std::vector<float> &img1,
                      const std::vector<float> &img2,
                      int w1, const std::vector<int> &w2,
                      std::vector<float> &blended);
    static std::vector<CurveParam> calcCurves(
            const std::vector<std::vector<float>> &vars, const int *spp,
            bool useLastTwoPoint = true);
    static int calcMinWeight(float v, float s, float i, int spp);

private:
    static std::vector<float> _jacobian(const std::vector<float> &denoised,
                                        const std::vector<float> &noisy,
                                        int w, int h,
                                        const std::vector<float> &var, float e,
                                        bool optiX, bool hdr, bool cleanAux);
    static CurveParam _leastSquares(const std::vector<float> &x,
                                    const std::vector<float> &y);
};

#endif // CURVEPREDICTOR_H
