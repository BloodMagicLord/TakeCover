#include <ViZDoom.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <thread>
#include <algorithm>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/viz/types.hpp>
#include <highgui.h>

void sleep(unsigned int time) {
    std::this_thread::sleep_for(std::chrono::milliseconds(time));
}

using namespace vizdoom;

using namespace cv;

int main() {
    Ptr<FeatureDetector> detector = ORB::create(10000);

    std::cout << "\n\nBASIC EXAMPLE\n\n";

    // Create DoomGame instance. It will run the game and communicate with you.
    auto *game = new DoomGame();
    game->loadConfig("basic2.cfg");
    game->init();


    // Define some actions. Each list entry corresponds to declared buttons:
    // MOVE_LEFT, MOVE_RIGHT, ATTACK
    // game.getAvailableButtonsSize() can be used to check the number of available buttons.
    // more combinations are naturally possible but only 3 are included for transparency when watching.
    std::vector<double> actions[game->getAvailableButtons().size()];
    /*for (int i = 0; i < game->getAvailableButtonsSize(); i++) {
        std::vector<double> action;
        for (int j = 0; j < game->getAvailableButtonsSize(); j++) {
            if (j == i)
                action.push_back(1);
            else
                action.push_back(0);
        }
        actions[i] = action;
    }*/

    actions[0] = {0, 1, 0};
    actions[1] = {1, 0, 0};
    actions[2] = {0, 0, 1};


    std::srand(time(0));

    // Run this many episodes
    int episodes = 2000;
    // Sets time that will pause the engine after each action.
    // Without this everything would go too fast for you to keep track of what's happening.
    unsigned int sleepTime = 2000 / DEFAULT_TICRATE; // = 28



    namedWindow("centers", WINDOW_AUTOSIZE);

    Mat now(game->getScreenHeight(), game->getScreenWidth(), CV_8UC3);
    Mat diff(game->getScreenHeight(), game->getScreenWidth(), CV_8UC3);
    Mat prev(game->getScreenHeight(), game->getScreenWidth(), CV_8UC3);

    const int d = 150;

    for (int i = 0; i < episodes; ++i) {

        std::cout << "Episode #" << i + 1 << "\n";

        // Starts a new episode. It is not needed right after init() but it doesn't cost much and the loop is nicer.
        game->newEpisode();

        while (!game->isEpisodeFinished()) {

            // Get the state
            GameStatePtr state = game->getState(); // GameStatePtr is std::shared_ptr<GameState>

            // Which consists of:
            unsigned int n = state->number;
            std::vector<double> vars = state->gameVariables;
            BufferPtr screenBuf = state->screenBuffer;
            BufferPtr depthBuf = state->depthBuffer;
            BufferPtr labelsBuf = state->labelsBuffer;
            BufferPtr automapBuf = state->automapBuffer;
            // BufferPtr is std::shared_ptr<Buffer> where Buffer is std::vector<uint8_t>
            std::vector<Label> labels = state->labels;

            prev = now.clone();

            for (int k = 0; k < now.rows; ++k) {
                for (int j = 0; j < now.cols; ++j) {
                    auto vectorCoord = 3 * (k * now.cols + j);

                    now.at<uchar>(k, 3 * j + 0) = (*screenBuf)[vectorCoord + 2];
                    now.at<uchar>(k, 3 * j + 1) = (*screenBuf)[vectorCoord + 1];
                    now.at<uchar>(k, 3 * j + 2) = (*screenBuf)[vectorCoord + 0];
                }
            }



            for (int k = 0; k < now.rows; ++k) {
                for (int j = 0; j < now.cols; ++j) {
                    int b = now.at<uchar>(k, 3 * j + 0);
                    int g = now.at<uchar>(k, 3 * j + 1);
                    int r = now.at<uchar>(k, 3 * j + 2);

                    int b1 = prev.at<uchar>(k, 3 * j + 0);
                    int g1 = prev.at<uchar>(k, 3 * j + 1);
                    int r1 = prev.at<uchar>(k, 3 * j + 2);

                    int B = abs(b - b1);
                    int G = abs(g - g1);
                    int R = abs(r - r1);

                    double A = sqrt(0.299 * r * r + 0.587 * g * g + 0.114 * b * b);

                    if (A > d) {
                        diff.at<uchar>(k, 3 * j + 0) = A;
                        diff.at<uchar>(k, 3 * j + 1) = A;
                        diff.at<uchar>(k, 3 * j + 2) = A;
                    } else {
                        diff.at<uchar>(k, 3 * j + 0) = 0;
                        diff.at<uchar>(k, 3 * j + 1) = 0;
                        diff.at<uchar>(k, 3 * j + 2) = 0;
                    }
                }
            }

            std::vector<Point> pts;
            std::vector<int> labls;
            cvtColor(diff, prev, COLOR_RGB2GRAY);

            findNonZero(prev, pts);

            double dst = 3, dst2 = dst * dst;
            int nLabels = cv::partition(pts, labls, [dst2](const Point& lhs, const Point& rhs) {
                return (hypot(lhs.x - rhs.x, lhs.y - rhs.y) < dst2);
            });

            std::vector<Point> middles(nLabels);
            std::vector<int> count(nLabels, 0);

            for (int k = 0; k < pts.size(); ++k) {
                count[labls[k]]++;
                middles[labls[k]].x += pts[k].x;
                middles[labls[k]].y += pts[k].y;
            }

            for (int k = 0; k < nLabels; ++k) {
                middles[k].x /= count[k];
                middles[k].y /= count[k];
            }



            for (int k = 0; k < nLabels; ++k) {

                if(middles[k].y > 0.56 * game->getScreenHeight()) {
                    middles.erase(middles.begin() + k--);
                    nLabels--;
                }
            }

            Mat m(game->getScreenHeight(), game->getScreenWidth(), CV_8UC3);
            for (int k = 0; k < nLabels; ++k) {
                circle(m, middles[k], 10, viz::Color(255,255,255));
            }

            imshow("centers", m);
            if(middles[0].x < game->getScreenWidth() / 2 - 10)
                game->makeAction(actions[1]);
            else if (middles[0].x > game->getScreenWidth() / 2 + 10)
                game->makeAction(actions[0]);
            else
                game->makeAction(actions[2]);



            /*std::vector<Vec3b> colors;
            for (int k = 0; k < nLabels; ++k) {
                colors.emplace_back(rand() & 255, rand() & 255, rand() & 255);
            }

            Mat3b lbl(game->getScreenHeight(), game->getScreenWidth(), Vec3b(0, 0, 0));
            for (int k = 0; k < pts.size(); ++k) {
                lbl(pts[k]) = colors[labls[k]];
            }

            imshow("diff", lbl);

            std::cout << nLabels << std::endl;*/

            waitKey(200);

            // Make random action and get reward

            // You can also get last reward by using this function
            // double reward = game->getLastReward();

            // Makes a "prolonged" action and skip frames.
            //int skiprate = 4
            //double reward = game.makeAction(choice(actions), skiprate)

            // The same could be achieved with:
            //game.setAction(choice(actions))
            //game.advanceAction(skiprate)
            //reward = game.getLastReward()

            //std::cout << "State #" << n << "\n";
            //std::cout << "Game variables: " << vars[0] << "\n";
            //std::cout << "Action reward: " << reward << "\n";
            //std::cout << "=====================\n";
        }

        std::cout << "Episode finished.\n";
        std::cout << "Total reward: " << game->getTotalReward() << "\n";
        std::cout << "************************\n";
    }
    cvDestroyWindow("name");
    // It will be done automatically in destructor but after close You can init it again with different settings.
    game->close();
    delete game;
}