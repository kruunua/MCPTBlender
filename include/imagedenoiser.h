/**
 * @file imagedenoiser.h
 * @author E. Denisova
 * @date 29/2/2024
 * @version 1.0
**/

#ifndef IMAGEDENOISER_H
#define IMAGEDENOISER_H

#include <vector>

class ImageDenoiser
{
private:
    ImageDenoiser();

public:
    static ImageDenoiser *instance();
    bool init();
    bool run(const std::vector<float> &input, int w, int h,
             std::vector<float> &output, bool optiX, bool hdr, bool cleanAux,
             bool cpu = false) const;
    void release();

private:
    bool _createOptiXContext();
    bool _createOptiXDenoiser(int idx);
    bool _runOptiX(const std::vector<float> &input, int w, int h,
                   std::vector<float> &output, bool hdr) const;
    static ImageDenoiser *m_instance;
    void *m_cpuData[3];
    void *m_gpuData[3];
    void *m_optiXData[3];
};

#endif // IMAGEDENOISER_H
