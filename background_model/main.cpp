#include "BackgroundGMM.h"
int main()
{
	using namespace std;
	string imageFolderPath = "..\\..\\KTL\\datastd\\";
	ostringstream oss;
	oss<<imageFolderPath+"image1.jpg";
	cv::Mat fr = cv::imread(oss.str()); // read the first frame
	BackgroundGMM bggmm(fr);
	for(int i = 1; i < 4500; i++)
	{
		oss.str("");
		oss<<imageFolderPath<<"image"<<i<<".jpg";
		cv::Mat img = cv::imread(oss.str());
		bggmm.Processing(img);
		bggmm.StoreImg(i);
		cout<<"\r"<<"               \r"<<i/4500.0f<<"% complete";
	}
	return 0;
}