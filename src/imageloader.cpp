/**
 * @file imageloader.cpp
 * @author E. Denisova
 * @date 29/2/2024
 * @version 1.0
**/

#include "imageloader.h"
#define IMATH_DLL

#include <ImfRgbaFile.h>
#include <ImfArray.h>
#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>

std::vector<float> ImageLoader::loadImage(const std::string &fileName,
                                          int &w, int &h)
{
    if(!std::ifstream(fileName))
        return std::vector<float>();

    try {
        Imf::RgbaInputFile file(fileName.c_str());
        Imath::Box2i dw = file.dataWindow();
        int width = dw.max.x - dw.min.x + 1;
        int height = dw.max.y - dw.min.y + 1;

        Imf::Array2D<Imf::Rgba> pixels(height, width);
        file.setFrameBuffer(&pixels[0][0] - dw.min.x - dw.min.y * width, 1,
                size_t(width));
        file.readPixels(dw.min.y, dw.max.y);

        size_t len = size_t(3 * width * height);
        std::vector<float> data(len);
        for(int i = 0; i < height; i++)
        {
            for(int j = 0; j < width; j++)
            {
                size_t idx = 3 * size_t(j + i * width);
                Imf::Rgba rgba = pixels[i][j];
                data[idx] = rgba.r;
                data[idx + 1] = rgba.g;
                data[idx + 2] = rgba.b;
            }
        }
        w = width;
        h = height;
        return data;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error reading " << fileName << ": " << e.what()
                  << std::endl;
        return std::vector<float>();
    }
}

std::vector<CurveParam> ImageLoader::loadCurves(const std::string &name0,
                                                const std::string &name1,
                                                int &w, int &h)
{
    std::vector<CurveParam> params;
    std::vector<float> slope = loadImage(name0, w, h);
    if(slope.empty())
        return params;

    std::vector<float> intercept = loadImage(name1, w, h);
    if(intercept.empty())
        return params;

    params.resize(size_t(3 * w * h));
    for(size_t i = 0; i < params.size(); i++)
    {
        params[i].first = slope[i];
        params[i].second = intercept[i];
    }
    return params;
}

std::vector<int> ImageLoader::loadWeights(const std::string &name, int w, int h)
{
    std::vector<int> weights;
    std::vector<float> weightsF = loadImage(name, w, h);
    if(weightsF.empty())
        return weights;

    weights.resize(size_t(3 * w * h));
    for(size_t i = 0; i < weights.size(); i++)
        weights[i] = int(std::round(weightsF[i] * 100000.f));

    return weights;
}

bool ImageLoader::saveExr(const std::vector<float> &data, int w, int h,
                          const std::string &name)
{
    const float *ptr = data.data();
    Imf::Array2D<Imf::Rgba> pixels(h, w);
    for(int y = 0; y < h; y++)
    {
        for(int x = 0; x < w; x++)
        {
            pixels[y][x] = Imf::Rgba(ptr[0], ptr[1], ptr[2]);
            ptr += 3;
        }
    }
    Imf::RgbaOutputFile file(name.c_str(), w, h, Imf::WRITE_RGB);
    file.setFrameBuffer(&pixels[0][0], 1, size_t(w));
    file.writePixels(h);
    return true;
}

bool ImageLoader::saveExr(const std::vector<CurveParam> &data, int w, int h,
                          const std::string &name0, const std::string &name1)
{
    const CurveParam *ptr = data.data();
    Imf::Array2D<Imf::Rgba> pixels0(h, w);
    Imf::Array2D<Imf::Rgba> pixels1(h, w);
    for(int y = 0; y < h; y++)
    {
        for(int x = 0; x < w; x++)
        {
            pixels0[y][x] = Imf::Rgba(ptr->first,
                                      (ptr + 1)->first,
                                      (ptr + 2)->first);
            pixels1[y][x] = Imf::Rgba(ptr->second,
                                      (ptr + 1)->second,
                                      (ptr + 2)->second);
            ptr += 3;
        }
    }
    Imf::RgbaOutputFile file0(name0.c_str(), w, h, Imf::WRITE_RGB);
    file0.setFrameBuffer(&pixels0[0][0], 1, size_t(w));
    file0.writePixels(h);
    Imf::RgbaOutputFile file1(name1.c_str(), w, h, Imf::WRITE_RGB);
    file1.setFrameBuffer(&pixels1[0][0], 1, size_t(w));
    file1.writePixels(h);
    return true;
}

bool ImageLoader::saveExr(const std::vector<int> &data, int w, int h,
                          const std::string &name)
{
    const int *ptr = data.data();
    Imf::Array2D<Imf::Rgba> pixels(h, w);
    for(int y = 0; y < h; y++)
    {
        for(int x = 0; x < w; x++)
        {
            pixels[y][x] = Imf::Rgba(
                        ptr[0] / 100000.f,
                        ptr[1] / 100000.f,
                        ptr[2] / 100000.f);
            ptr += 3;
        }
    }
    Imf::RgbaOutputFile file(name.c_str(), w, h, Imf::WRITE_RGB);
    file.setFrameBuffer(&pixels[0][0], 1, size_t(w));
    file.writePixels(h);
    return true;
}

namespace {
template<typename T>
constexpr const T &clamp(const T &v, const T &lo, const T &hi)
{
    return (v < lo) ? lo : (hi < v) ? hi : v;
}
}

void ImageLoader::gaussianBlur(const std::vector<float> &src,
                               std::vector<float> &dst,
                               int w, int h, int kernelSize,
                               const std::vector<float> &var)
{
    float mean = avg(var);
    float sigma = std::max(std::sqrt(mean), 1e-3f) * 100;
    double s = double(sigma);
    int halfKernel = kernelSize / 2;

    double aux0 = 2 * s * s;
    double aux1 = M_PI * aux0;
    std::vector<float> kernel(size_t(kernelSize * kernelSize));
    float *kPtr = kernel.data();
    float weightSum = 0;
    for(int ky = -halfKernel; ky <= halfKernel; ky++)
    {
        for(int kx = -halfKernel; kx <= halfKernel; kx++)
        {
            *kPtr = float(std::exp(-(kx * kx + ky * ky) / aux0) / aux1);
            weightSum += *kPtr++;
        }
    }
    std::vector<float> res(src.size());
    for(int y = 0; y < h; y++)
    {
        for(int x = 0; x < w; x++)
        {
            float dstR = 0, dstG = 0, dstB = 0;
            float *kPtr = kernel.data();
            for(int ky = -halfKernel; ky <= halfKernel; ky++)
            {
                for(int kx = -halfKernel; kx <= halfKernel; kx++)
                {
                    float weight = *kPtr++;
                    int pixelX = clamp(x + kx, 0, w - 1);
                    int pixelY = clamp(y + ky, 0, h - 1);
                    size_t idx = size_t(pixelX + pixelY * w) * 3;
                    float r = src[idx];
                    float g = src[idx + 1];
                    float b = src[idx + 2];
                    if(!std::isnormal(r))
                        r = 0;
                    if(!std::isnormal(g))
                        g = 0;
                    if(!std::isnormal(b))
                        b = 0;

                    dstR += weight * r;
                    dstG += weight * g;
                    dstB += weight * b;
                }
            }
            size_t dstIdx = size_t(x + y * w) * 3;
            res[dstIdx] = dstR / weightSum;
            res[dstIdx + 1] = dstG / weightSum;
            res[dstIdx + 2] = dstB / weightSum;
        }
    }
    dst = res;
}

float ImageLoader::mse(const std::vector<float> &img1,
                       const std::vector<float> &img2)
{
    size_t len = std::min(img1.size(), img2.size());
    std::vector<float> aux(len);
    std::transform(img1.begin(), img1.begin() + int(len), img2.begin(),
                   aux.begin(), [](float a, float b) {
                       if(!std::isnormal(a)) a = 0;
                       if(!std::isnormal(b)) b = 0;
                       return (a - b) * (a - b);
                   });
    return avg(aux);
}

float ImageLoader::avg(const std::vector<float> &img)
{
    float s = 0;
    for(float val : img)
        s += std::isnormal(val) ? val : 0.0f;

    return s / img.size();
}

std::vector<float> ImageLoader::mseVector(const std::vector<float> &img1,
                                          const std::vector<float> &img2)
{
    size_t len = std::min(img1.size(), img2.size());
    std::vector<float> aux(len);
    std::transform(img1.begin(), img1.begin() + int(len), img2.begin(),
                   aux.begin(), [](float a, float b) {
                       if(!std::isnormal(a)) a = 0;
                       if(!std::isnormal(b)) b = 0;
                       return (a - b) * (a - b);
                   });
    return aux;
}

std::vector<float> ImageLoader::diff(const std::vector<float> &img,
                                     const std::vector<float> &ref)
{
    float mean = 0.05f;
    std::vector<float> diffVec(std::min(img.size(), ref.size()), 0);
    for(size_t i = 0; i < diffVec.size(); i += 3)
    {
        float rdiff = img[i] - ref[i];
        float gdiff = img[i + 1] - ref[i + 1];
        float bdiff = img[i + 2] - ref[i + 2];
        float d = rdiff * rdiff + gdiff * gdiff + bdiff * bdiff;
        diffVec[i] = d / mean;
        diffVec[i + 1] = d / mean;
        diffVec[i + 2] = d / mean;
    }
    return diffVec;
}
