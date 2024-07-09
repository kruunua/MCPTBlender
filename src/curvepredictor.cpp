/**
 * @file curvepredictor.cpp
 * @author E. Denisova
 * @date 29/2/2024
 * @version 1.0
**/

#include "curvepredictor.h"
#include "imagedenoiser.h"
#include "imageloader.h"
#include <algorithm>
#include <random>

std::vector<float> CurvePredictor::sure(const std::vector<float> &denoised,
                                        const std::vector<float> &noisy,
                                        int w, int h,
                                        const std::vector<float> &var,
                                        bool useOptiX, bool hdr, bool cleanAux)
{
    const int tries = 1;
    const float e = 1;
    std::vector<float> jacob(denoised.size(), 0.0f);

    for(int i = 0; i < tries; i++)
    {
        std::vector<float> jacob0 = _jacobian(denoised, noisy, w, h, var, e,
                                              useOptiX, hdr, cleanAux);
        std::transform(jacob.begin(), jacob.end(), jacob0.begin(),
                       jacob.begin(), std::plus<float>());
    }
    std::vector<float> mse = ImageLoader::mseVector(denoised, noisy);
    for(size_t i = 0; i < jacob.size(); i++)
    {
        float j = 2 * jacob[i] / tries;
        float v = std::isnormal(var[i]) ? var[i] : 0.f;
        jacob[i] = mse[i] - v + j;
    }
    return jacob;
}

int CurvePredictor::denoisedWeight(float sure, const CurveParam &outs,
                                   int minWeight)
{
    const int limit = 65536;
    if(sure <= 1e-12f)
        return std::min(minWeight, limit);

    CurveParam cp(outs);

    if(outs.first < 1e-6f && outs.second < 1e-6f)
        return std::min(minWeight, limit);

    float sureLog = std::log(sure);
    float x = limit;
    const float c = 100;
    x = std::pow((sureLog + c) / cp.first, 1.0f / cp.second);

    if(x >= limit)
        return limit;

    return int(x);
}

void CurvePredictor::blend(const std::vector<float> &img1,
                           const std::vector<float> &img2,
                           int w1, const std::vector<int> &w2,
                           std::vector<float> &blended)
{
    size_t len = std::min(img1.size(), img2.size());
    blended.resize(len);
    std::transform(img2.begin(), img2.begin() + int(len), w2.begin(),
                   blended.begin(), [](float a, int b) {
                       return a * b;
                   });
    std::transform(img1.begin(), img1.begin() + int(len), blended.begin(),
                   blended.begin(), [w1](float a, float b) {
                       return a * w1 + b;
                   });
    std::transform(w2.begin(), w2.begin() + int(len), blended.begin(),
                   blended.begin(), [w1](int a, float b) {
                       return b / (w1 + a);
                   });
}

std::vector<CurveParam> CurvePredictor::calcCurves(
        const std::vector<std::vector<float> > &vars, const int *spp,
        bool useLastTwoPoint)
{
    std::vector<CurveParam> params(vars[0].size(), CurveParam(.0f, .0f));
    if(vars.size() < 2)
        return params;

    for(size_t i = 0; i < params.size(); i++)
    {
        std::vector<float> vals;
        for(size_t j = 0; j < vars.size(); j++)
        {
            float v = vars[j][i];
            vals.push_back(std::isnormal(v) ? v : 0.0f);
        }
        size_t idx0 = 1;
        std::vector<float> goodVals;
        for(size_t j = vals.size() - 1; j > 0; j--)
        {
            if(vals[j] < vals[j - 1])
            {
                if(vals[j] > 0)
                {
                    goodVals.insert(goodVals.begin(), vals[j]);
                    if(useLastTwoPoint && goodVals.size() == 2)
                    {
                        idx0 = j;
                        break;
                    }
                }
            }
            else
            {
                goodVals.insert(goodVals.begin(), vals[j]);
                idx0 = j;
                break;
            }
        }
        size_t idx1 = goodVals.size() - 1;
        size_t len = idx1 - idx0 + 1;
        if(int(len) < 2)
            continue;

        std::vector<float> x(len);
        std::vector<float> y(len);
        const float c = 100;
        for(size_t k = idx0; k <= idx1; k++)
        {
            float A = std::log(float(spp[k]));
            float B = std::log(goodVals[k - idx0]);
            x[k - idx0] = 1.0f / A;
            y[k - idx0] = std::log(B + c) / A;
        }
        CurveParam slopeIntercept = _leastSquares(x, y);
        float b = slopeIntercept.second;
        float a = slopeIntercept.first;
        a = std::exp(a);

        params[i] = CurveParam(a, b);
    }
    return params;
}

int CurvePredictor::calcMinWeight(float v, float s, float i, int spp)
{
    const float e = 1e-7f;

    if(v < e && i < e)
        return 65536;

    float expN = std::pow(10.0f, 20);
    float sqrtV = std::sqrt(v + e) / (expN / spp - 1);
    float expV = sqrtV * sqrtV;
    float a = (std::log(v + e) - std::log(expV))
            / (std::log(float(spp)) - std::log(expN));
    float b = std::log(v + e) - a * std::log(float(spp));
    float minWeight = std::exp(std::log(float(spp)) * (std::log(s + e) - b)
                               / (std::log(v + e) - b));
    return std::min(int(std::round(minWeight)), 65536);
}

std::vector<float> CurvePredictor::_jacobian(const std::vector<float> &denoised,
                                             const std::vector<float> &noisy,
                                             int w, int h,
                                             const std::vector<float> &var,
                                             float e, bool optiX, bool hdr,
                                             bool cleanAux)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> std_nrm(0.f, 1.f);

    std::vector<float> fy(denoised);
    std::vector<float> fz(denoised.size());
    std::vector<float> b(denoised.size());
    std::vector<float> z(noisy);

    for(size_t i = 0; i < denoised.size(); i++)
    {
        float v = std::isnormal(var[i]) ? var[i] : 0.f;
        b[i] = std_nrm(gen) * std::sqrt(v);
        z[i] = noisy[i] + e * b[i];
    }
    if(!ImageDenoiser::instance()->run(z, w, h, fz, optiX, hdr, cleanAux))
        return std::vector<float>();

    std::vector<float> res(denoised.size(), 0);
    for(size_t i = 0; i < fy.size(); i++)
        res[i] = b[i] / e * (fz[i] - fy[i]);

    return res;
}

CurveParam CurvePredictor::_leastSquares(const std::vector<float> &x,
                                         const std::vector<float> &y)
{
    const size_t n = x.size();
    float sum_x = 0, sum_y = 0, sum_xy = 0, sum_x_squared = 0;
    for(size_t i = 0; i < n; i++)
    {
        sum_x += x[i];
        sum_y += y[i];
        sum_xy += x[i] * y[i];
        sum_x_squared += x[i] * x[i];
    }
    float mean_x = sum_x / n;
    float mean_y = sum_y / n;
    float slope = (n * sum_xy - sum_x * sum_y)
            / (n * sum_x_squared - sum_x * sum_x);
    float intercept = mean_y - slope * mean_x;
    return CurveParam(slope, intercept);
}
