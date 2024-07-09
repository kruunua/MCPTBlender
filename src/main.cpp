/**
 * @file main.cpp
 * @author E. Denisova
 * @date 29/2/2024
 * @version 1.0
 *
**/
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include "imageloader.h"
#include "curvepredictor.h"
#include "imagedenoiser.h"

int main(int argc, char *argv[])
{
    bool useAlbedo = true;
    bool useNormal = true;
    bool applyGB = true;
    bool useOptiX = false;
    int denoiseUntil = -1;
    bool recalcAll = false;
    // Gaussian Blur
    const int winSize = 11;
    if(argc < 2)
    {
        std::cout << "[MCPTBlender] <PATH_TO_HDR>" << std::endl;
        goto help;
    }
    for(int i = 2; i < argc; i++)
    {
        if(std::string(argv[i]) == "/?")
        {
help:
            std::cout << "   -x          use optiX (default OIDN)"
                      << std::endl;
            std::cout << "   -a-         do not use albedo+normal "
                         "(default true)" << std::endl;
            std::cout << "   -n-         do not use normal "
                         "(default true)" << std::endl;
            std::cout << "   -o          apply oidn on estimates "
                         "(default Gaussian Blur)" << std::endl;
            std::cout << "   -u N        denoise until N "
                         "(default last)" << std::endl;
            std::cout << "   -c          recalculate all "
                         "(default read from file if exists)" << std::endl;
            std::cout << "   /?          show this help" << std::endl;
            return 0;
        }
        if(std::string(argv[i]) == "-x")
            useOptiX = true;
        else if(std::string(argv[i]) == "-a-")
        {
            useAlbedo = false;
            useNormal = false;
        }
        else if(std::string(argv[i]) == "-n-")
            useNormal = false;
        else if(std::string(argv[i]) == "-o")
            applyGB = false;
        else if(std::string(argv[i]) == "-u" && i < argc - 1)
            denoiseUntil = std::stoi(argv[i + 1]);
        else if(std::string(argv[i]) == "-c")
            recalcAll = true;
    }
    std::string path(argv[1]);
    std::vector<std::experimental::filesystem::path> files;
    for(const auto &entry : std::experimental::filesystem::directory_iterator(path))
    {
        if(entry.status().type() == std::experimental::filesystem::file_type::regular)
        {
            std::string filename = entry.path().filename().string();
            if(filename.find("spp.hdr.exr") == filename.length() - 11)
                files.push_back(entry.path());
        }
    }
    if(files.empty())
    {
        std::cout << "No HDR found!" << std::endl;
        return -1;
    }
    std::string fileName = files.back().filename().string();
    size_t c = fileName.length() - 18; // name_NNNNNNspp
    fileName = fileName.substr(0, c);
    std::vector<int> spp;
    for(auto file : files)
    {
        std::string baseName = file.filename().string();
        if(baseName.substr(0, c) != fileName)
            continue;

        std::string n = baseName.substr(baseName.length() - 17);
        spp.push_back(std::stoi(n));
    }
    if(denoiseUntil == -1 || std::find(spp.begin(), spp.end(), denoiseUntil) == spp.end())
        denoiseUntil = spp.back();

    int w = 0, h = 0;
    // 1. Read REF
    std::string refPath = files.back().string();
    std::vector<float> ref = ImageLoader::loadImage(refPath, w, h);

    std::string denAlg = useOptiX ? "OptiX" : "OIDN";
    std::cout << "\tOURS\t\t" << denAlg << "\t\tMC" << std::endl;

    size_t len = spp.size();
    std::vector<std::vector<float>> varsVec;
    for(size_t i = 0; i < len; i++)
    {
        int denNo = std::min(spp[i], denoiseUntil);
        std::string sppStr = std::to_string(spp[i]);
        sppStr.insert(0, 6 - sppStr.length(), '0');
        std::string denNoStr = std::to_string(denNo);
        denNoStr.insert(0, 6 - denNoStr.length(), '0');

        // 2. Read VAR
        std::string varPath = path + "/" + fileName + "_" + sppStr
                + "spp.var.exr";
        std::vector<float> var = ImageLoader::loadImage(varPath, w, h);
        if(var.empty())
        {
            std::cout << "Error loading " << varPath << std::endl;
            continue;
        }
        // 3. Read HDR
        std::string imgPath = path + "/" + fileName + "_" + sppStr
                + "spp.hdr.exr";
        std::vector<float> img = ImageLoader::loadImage(imgPath, w, h);
        if(img.empty())
        {
            std::cout << "Error loading " << imgPath << std::endl;
            continue;
        }
        // 4. Filter VAR
        if(applyGB)
        {
            std::string varGaussPath = path + "/" + fileName + "_" + sppStr
                    + "spp.var.gb.exr";
            std::vector<float> gaussVar;
            if(!recalcAll)
                gaussVar = ImageLoader::loadImage(varGaussPath, w, h);

            if(gaussVar.empty())
            {
                ImageLoader::gaussianBlur(var, gaussVar, w, h, winSize, var);
                ImageLoader::saveExr(gaussVar, w, h, varGaussPath);
            }
            varsVec.push_back(gaussVar);
        }
        else
        {
            std::string varOidnPath = path + "/" + fileName + "_" + sppStr
                    + "spp.var.oidn.exr";
            std::vector<float> oidnVar;
            if(!recalcAll)
                oidnVar = ImageLoader::loadImage(varOidnPath, w, h);

            if(oidnVar.empty())
            {
                ImageDenoiser::instance()->init();
                ImageDenoiser::instance()->run(var, w, h, oidnVar, useOptiX,
                                               true, true);
                ImageLoader::saveExr(oidnVar, w, h, varOidnPath);
            }
            varsVec.push_back(oidnVar);
        }
        std::vector<float> inputImg = img;
        std::vector<float> inputVar = var;

        // 5. If denoising stopped, read correct HDR and VAR for DEN/SURE
        if(spp[i] != denNo)
        {
            std::string imgPath = path + "/" + fileName + "_" + denNoStr
                    + "spp.hdr.exr";
            inputImg = ImageLoader::loadImage(imgPath, w, h);
            if(inputImg.empty())
            {
                std::cout << "Error loading " << imgPath << std::endl;
                continue;
            }
            std::string varPath = path + "/" + fileName + "_" + denNoStr
                    + "spp.var.exr";
            inputVar = ImageLoader::loadImage(varPath, w, h);
            if(inputVar.empty())
            {
                std::cout << "Error loading " << varPath << std::endl;
                continue;
            }
        }
        // 6. Read DEN
        std::string denPath = path + "/" + fileName + "_" + denNoStr
                + "spp." + (useOptiX ? "optix" : "oidn")
                + (useNormal ? "_alb_nrm" : useAlbedo ? "_alb" : "")
                + ".exr";
        std::vector<float> denoised;
        if(!recalcAll)
            denoised = ImageLoader::loadImage(denPath, w, h);

        // ...and SURE
        std::string surePath = path + "/" + fileName + "_" + denNoStr
                + "spp." + (useOptiX ? "optix" : "oidn")
                + (useNormal ? "_alb_nrm" : useAlbedo ? "_alb" : "")
                + ".sure.exr";
        std::vector<float> sure;
        if(!recalcAll)
            sure = ImageLoader::loadImage(surePath, w, h);

        // 7. If DEN/SURE not read correctly, calculate
        if(denoised.empty() || sure.empty())
        {
            std::vector<float> alb, nor;
            if(useAlbedo)
            {
                std::string albPath = path + "/" + fileName + "_"
                        + denNoStr + "spp.alb.exr";
                alb = ImageLoader::loadImage(albPath, w, h);
                useAlbedo = !alb.empty();
                if(useAlbedo && useNormal)
                {
                    std::string norPath = path + "/" + fileName + "_"
                            + denNoStr + "spp.nrm.exr";
                    nor = ImageLoader::loadImage(norPath, w, h);
                    useNormal = !nor.empty();
                }
            }
            if(useAlbedo)
            {
                inputImg.resize(6 * size_t(w * h));
                size_t len = 3 * size_t(w * h) * sizeof(float);
                memcpy(inputImg.data() + 3 * w * h, alb.data(), len);
                if(useNormal)
                {
                    inputImg.resize(9 * size_t(w * h));
                    memcpy(inputImg.data() + 6 * w * h, nor.data(), len);
                }
            }
            ImageDenoiser::instance()->init();
            ImageDenoiser::instance()->run(inputImg, w, h, denoised, useOptiX,
                                           true, false);
            ImageLoader::saveExr(denoised, w, h, denPath);

            ImageDenoiser::instance()->init();
            sure = CurvePredictor::sure(denoised, inputImg, w, h, inputVar,
                                        useOptiX, true, false);
            ImageLoader::saveExr(sure, w, h, surePath);
        }
        // 8. Filter SURE
        std::vector<float> filteredSure;
        if(applyGB)
        {
            std::string filteredPath = path + "/" + fileName + "_"
                    + denNoStr + "spp." + (useOptiX ? "optix" : "oidn")
                    + (useNormal ? "_alb_nrm" : useAlbedo ? "_alb" : "")
                    + ".sure.gb.exr";
            if(!recalcAll)
                filteredSure = ImageLoader::loadImage(filteredPath, w, h);

            if(filteredSure.empty())
            {
                ImageLoader::gaussianBlur(sure, filteredSure, w, h, winSize,
                                          inputVar);
                ImageLoader::saveExr(filteredSure, w, h, filteredPath);
            }
        }
        else
        {
            std::string filteredPath = path + "/" + fileName + "_"
                    + denNoStr + "spp." + (useOptiX ? "optix" : "oidn")
                    + (useNormal ? "_alb_nrm" : useAlbedo ? "_alb" : "")
                    + ".sure.oidn.exr";
            if(!recalcAll)
                filteredSure = ImageLoader::loadImage(filteredPath, w, h);

            if(filteredSure.empty())
            {
                ImageDenoiser::instance()->init();
                ImageDenoiser::instance()->run(sure, w, h, filteredSure,
                                               useOptiX, true, true);
                ImageLoader::saveExr(filteredSure, w, h, filteredPath);
            }
        }
        float avgSure = ImageLoader::avg(sure);
        float avgVar = ImageLoader::avg(var);

        std::vector<float> filteredVar = varsVec.back();
        // 9. If OIDN for estimates, stop filtering if avgSure > avgVar
        if(!applyGB && avgSure > avgVar)
        {
            filteredSure = sure;
            filteredVar = var;
        }
        // 10. Calculate curves on-the-fly
        std::string slopePath = path + "/" + fileName + "_" + sppStr
                + "spp.slope.exr";
        std::string interceptPath = path + "/" + fileName + "_" + sppStr
                + "spp.intercept.exr";
        std::vector<CurveParam> curveParams = CurvePredictor::calcCurves(
                    varsVec, spp.data());
        ImageLoader::saveExr(curveParams, w, h, slopePath, interceptPath); // For debug

        // 11. Calculate weights
        std::string weightsPath = path + "/" + fileName + "_" + sppStr
                + "spp.weights.exr";
        std::vector<int> weights(var.size(), 0);
        for(size_t j = 0; j < weights.size(); j++)
        {
            float s = filteredSure[j];
            float v = filteredVar[j];

            if(!std::isnormal(s)) s = 0;
            if(!std::isnormal(v)) v = 0;

            // 11a. If avgVar > avgSure set negative SURE to 0;
            // otherwise, to magnitude
            if(avgVar > avgSure)
                s = std::max(s, 0.0f);
            else
                s = std::abs(s);

            // 11b. Min. weight for DEN
            int minWgh = CurvePredictor::calcMinWeight(v, s, img[j], spp[i]);
            // 11c. Weight
            weights[j] = CurvePredictor::denoisedWeight(s, curveParams[j],
                                                        minWgh);
        }
        ImageLoader::saveExr(weights, w, h, weightsPath); // For debug
        // 12. Blending
        std::vector<float> blended;
        CurvePredictor::blend(img, denoised, spp[i], weights, blended);
        float b = ImageLoader::mse(blended, ref);
        float d = ImageLoader::mse(denoised, ref);
        float n = ImageLoader::mse(img, ref);
        std::cout << sppStr << "\t" << b << "\t" << d << "\t" << n
                  << std::endl;

        std::string bndPath = path + "/" + fileName + "_" + sppStr
                + "spp.ours." + (useOptiX ? "optix" : "oidn")
                + (useNormal ? "_alb_nrm" : useAlbedo ? "_alb" : "")
                + (applyGB ? ".gb" : ".oidn") + ".exr";
        ImageLoader::saveExr(blended, w, h, bndPath);

        std::string diffPath = path + "/" + fileName + "_" + sppStr
                + "spp.ours." + (useOptiX ? "optix" : "oidn")
                + (useNormal ? "_alb_nrm" : useAlbedo ? "_alb" : "")
                + (applyGB ? ".gb" : ".oidn") + ".diff.exr";
        std::vector<float> diff = ImageLoader::diff(blended, ref);
        ImageLoader::saveExr(diff, w, h, diffPath);

        diffPath = path + "/" + fileName + "_" + sppStr + "spp."
                + (useOptiX ? "optix" : "oidn")
                + (useNormal ? "_alb_nrm" : useAlbedo ? "_alb" : "")
                + ".diff.exr";
        diff = ImageLoader::diff(denoised, ref);
        ImageLoader::saveExr(diff, w, h, diffPath);
    }
    std::cout << "All done" << std::endl;
    ImageDenoiser::instance()->release();
    return 0;
}
