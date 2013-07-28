#pragma once
#include <opencv2\opencv.hpp>

struct Rank
{
	float val;
	int index;
};

class BackgroundGMM
{
private:
	const int C; //# of Gaussian components
	const int M; //# of background components
	const float D; //positive deviation threshold
	const float alpha; //learning rate
	const float thresh; //foreground threshold
	const float sd_init; //initial std for new components
	cv::Mat fr;
	cv::Mat fr_bw; //gray scale frame
	cv::Mat fg; //foreground for each frame
	cv::Mat fg_mask; //foreground map for each pixel
	cv::Mat bg_bw; //background pixel
	int width;
	int height;//width,height
	float *w; //weight array
	float *mean; //pixel means
	float *sd; //pixel std
	float *u_diff; //difference of each pixel from mean
	float p; //initial p variable (used to update mean and std)
	void ProcessingOnePix(int i, int j);//Returns true if the pixel <i,j> is considered as foreground
	void PreInit(const cv::Mat &frame, int n);

public:
	BackgroundGMM(const cv::Mat &first_frame, bool repeat = true, int n = 5);
	~BackgroundGMM(void);
	void Processing(const cv::Mat &frame);
	bool DetermineForeground(int i, int j); //whether ith row, jth col pixel foreground
	void StoreImg(int frameNO);
};

