
    This file is part of Leela Zero.
    Copyright (C) 2017 Gian-Carlo Pascutto
    Leela Zero is free software you can redistribute it andor modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    Leela Zero is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with Leela Zero.  If not, see httpwww.gnu.orglicenses.



#include config.h
#include Network.h

#include algorithm
#include array
#include cassert
#include cmath
#include iterator
#include memory
#include sstream
#include string
#include boostutility.hpp
#include boostformat.hpp
#include boostspirithomex3.hpp

#ifdef __APPLE__
#include AccelerateAccelerate.h
#endif
#ifdef USE_MKL
#include mkl.h
#endif
#ifdef USE_OPENBLAS
#include cblas.h
#endif
#ifdef USE_OPENCL
#include OpenCLScheduler.h
#include UCTNode.h
#endif

#include FastBoard.h
#include FastState.h
#include FullBoard.h
#include GameState.h
#include GTP.h
#include Im2Col.h
#include NNCache.h
#include Random.h
#include ThreadPool.h
#include Timing.h
#include Utils.h

namespace x3 = boostspiritx3;
using namespace Utils;

 Input + residual block tower
static stdvectorstdvectorfloat conv_weights;
static stdvectorstdvectorfloat conv_biases;
static stdvectorstdvectorfloat batchnorm_means;
static stdvectorstdvectorfloat batchnorm_stddivs;

 Policy head
static stdvectorfloat conv_pol_w;
static stdvectorfloat conv_pol_b;
static stdarrayfloat, 2 bn_pol_w1;
static stdarrayfloat, 2 bn_pol_w2;

static stdarrayfloat, (BOARD_SQUARES + 1)  BOARD_SQUARES  2 ip_pol_w;
static stdarrayfloat, BOARD_SQUARES + 1 ip_pol_b;

 Value head
static stdvectorfloat conv_val_w;
static stdvectorfloat conv_val_b;
static stdarrayfloat, 1 bn_val_w1;
static stdarrayfloat, 1 bn_val_w2;

static stdarrayfloat, BOARD_SQUARES  256 ip1_val_w;
static stdarrayfloat, 256 ip1_val_b;

static stdarrayfloat, 256 ip2_val_w;
static stdarrayfloat, 1 ip2_val_b;

 Rotation helper
static stdarraystdarrayint, BOARD_SQUARES, 8 rotate_nn_idx_table;

void Networkbenchmark(const GameState const state, const int iterations) {
    const auto cpus = cfg_num_threads;
    const Time start;

    ThreadGroup tg(thread_pool);
    stdatomicint runcount{0};

    for (auto i = 0; i  cpus; i++) {
        tg.add_task([&runcount, iterations, state]() {
            while (runcount  iterations) {
                runcount++;
                get_scored_moves(state, EnsembleRANDOM_ROTATION, -1, true);
            }
        });
    }
    tg.wait_all();

    const Time end;
    const auto elapsed = Timetimediff_seconds(start, end);
    myprintf(%5d evaluations in %5.2f seconds - %d nsn,
             runcount.load(), elapsed, int(runcount.load()  elapsed));
}

void Networkprocess_bn_var(stdvectorfloat& weights, const float epsilon) {
    for(auto&& w  weights) {
        w = 1.0f  stdsqrt(w + epsilon);
    }
}

stdvectorfloat Networkwinograd_transform_f(const stdvectorfloat& f,
                                                 const int outputs,
                                                 const int channels) {
     F(2x2, 3x3) Winograd filter transformation
     transpose(G.dot(f).dot(G.transpose()))
     U matrix is transposed for better memory layout in SGEMM
    auto U = stdvectorfloat(WINOGRAD_TILE  outputs  channels);
    const auto G = stdarrayfloat, WINOGRAD_TILE{ 1.0,  0.0,  0.0,
                                                     0.5,  0.5,  0.5,
                                                     0.5, -0.5,  0.5,
                                                     0.0,  0.0,  1.0};
    auto temp = stdarrayfloat, 12{};

    for (auto o = 0; o  outputs; o++) {
        for (auto c = 0; c  channels; c++) {
            for (auto i = 0; i  4; i++){
                for (auto j = 0; j  3; j++) {
                    auto acc = 0.0f;
                    for (auto k = 0; k  3; k++) {
                        acc += G[i3 + k]  f[ochannels9 + c9 + k3 + j];
                    }
                    temp[i3 + j] = acc;
                }
            }

            for (auto xi = 0; xi  4; xi++) {
                for (auto nu = 0; nu  4; nu++) {
                    auto acc = 0.0f;
                    for (auto k = 0; k  3; k++) {
                        acc += temp[xi3 + k]  G[nu3 + k];
                    }
                    U[xi  (4  outputs  channels)
                      + nu  (outputs  channels)
                      + c  outputs
                      + o] = acc;
                }
            }
        }
    }

    return U;
}

stdvectorfloat Networkzeropad_U(const stdvectorfloat& U,
                                      const int outputs, const int channels,
                                      const int outputs_pad,
                                      const int channels_pad) {
     Fill with zeroes
    auto Upad = stdvectorfloat(WINOGRAD_TILE  outputs_pad  channels_pad);

    for(auto o = 0; o  outputs; o++) {
        for(auto c = 0; c  channels; c++) {
            for(auto xi = 0; xi  WINOGRAD_ALPHA; xi++){
                for(auto nu = 0; nu  WINOGRAD_ALPHA; nu++) {
                    Upad[xi  (WINOGRAD_ALPHA  outputs_pad  channels_pad)
                         + nu  (outputs_pad  channels_pad)
                         + c  outputs_pad +
                          o] =
                    U[xi  (WINOGRAD_ALPHA  outputs  channels)
                      + nu  (outputs  channels)
                      + c  outputs
                      + o];
                }
            }
        }
    }

    return Upad;
}

stdpairint, int Networkload_v1_network(stdifstream& wtfile) {
     Count size of the network
    myprintf(Detecting residual layers...);
     We are version 1
    myprintf(v%d..., 1);
     First line was the version number
    auto linecount = size_t{1};
    auto channels = 0;
    auto line = stdstring{};
    while (stdgetline(wtfile, line)) {
        auto iss = stdstringstream{line};
         Third line of parameters are the convolution layer biases,
         so this tells us the amount of channels in the residual layers.
         We are assuming all layers have the same amount of filters.
        if (linecount == 2) {
            auto count = stddistance(stdistream_iteratorstdstring(iss),
                                       stdistream_iteratorstdstring());
            myprintf(%d channels..., count);
            channels = count;
        }
        linecount++;
    }
     1 format id, 1 input layer (4 x weights), 14 ending weights,
     the rest are residuals, every residual has 8 x weight lines
    auto residual_blocks = linecount - (1 + 4 + 14);
    if (residual_blocks % 8 != 0) {
        myprintf(nInconsistent number of weights in the file.n);
        return {0, 0};
    }
    residual_blocks = 8;
    myprintf(%d blocks.n, residual_blocks);

     Re-read file and process
    wtfile.clear();
    wtfile.seekg(0, stdiosbeg);

     Get the file format id out of the way
    stdgetline(wtfile, line);

    const auto plain_conv_layers = 1 + (residual_blocks  2);
    const auto plain_conv_wts = plain_conv_layers  4;
    linecount = 0;
    while (stdgetline(wtfile, line)) {
        stdvectorfloat weights;
        auto it_line = line.cbegin();
        const auto ok = phrase_parse(it_line, line.cend(),
                                     x3float_, x3space, weights);
        if (!ok  it_line != line.cend()) {
            myprintf(nFailed to parse weight file. Error on line %d.n,
                    linecount + 2); +1 from version line, +1 from 0-indexing
            return {0,0};
        }
        if (linecount  plain_conv_wts) {
            if (linecount % 4 == 0) {
                conv_weights.emplace_back(weights);
            } else if (linecount % 4 == 1) {
                 Redundant in our model, but they encode the
                 number of outputs so we have to read them in.
                conv_biases.emplace_back(weights);
            } else if (linecount % 4 == 2) {
                batchnorm_means.emplace_back(weights);
            } else if (linecount % 4 == 3) {
                process_bn_var(weights);
                batchnorm_stddivs.emplace_back(weights);
            }
        } else if (linecount == plain_conv_wts) {
            conv_pol_w = stdmove(weights);
        } else if (linecount == plain_conv_wts + 1) {
            conv_pol_b = stdmove(weights);
        } else if (linecount == plain_conv_wts + 2) {
            stdcopy(cbegin(weights), cend(weights), begin(bn_pol_w1));
        } else if (linecount == plain_conv_wts + 3) {
            process_bn_var(weights);
            stdcopy(cbegin(weights), cend(weights), begin(bn_pol_w2));
        } else if (linecount == plain_conv_wts + 4) {
            stdcopy(cbegin(weights), cend(weights), begin(ip_pol_w));
        } else if (linecount == plain_conv_wts + 5) {
            stdcopy(cbegin(weights), cend(weights), begin(ip_pol_b));
        } else if (linecount == plain_conv_wts + 6) {
            conv_val_w = stdmove(weights);
        } else if (linecount == plain_conv_wts + 7) {
            conv_val_b = stdmove(weights);
        } else if (linecount == plain_conv_wts + 8) {
            stdcopy(cbegin(weights), cend(weights), begin(bn_val_w1));
        } else if (linecount == plain_conv_wts + 9) {
            process_bn_var(weights);
            stdcopy(cbegin(weights), cend(weights), begin(bn_val_w2));
        } else if (linecount == plain_conv_wts + 10) {
            stdcopy(cbegin(weights), cend(weights), begin(ip1_val_w));
        } else if (linecount == plain_conv_wts + 11) {
            stdcopy(cbegin(weights), cend(weights), begin(ip1_val_b));
        } else if (linecount == plain_conv_wts + 12) {
            stdcopy(cbegin(weights), cend(weights), begin(ip2_val_w));
        } else if (linecount == plain_conv_wts + 13) {
            stdcopy(cbegin(weights), cend(weights), begin(ip2_val_b));
        }
        linecount++;
    }
    wtfile.close();

    return {channels, residual_blocks};
}

stdpairint, int Networkload_network_file(const stdstring& filename) {
    auto wtfile = stdifstream{filename};
    if (wtfile.fail()) {
        myprintf(Could not open weights file %sn, filename.c_str());
        return {0, 0};
    }

     Read format version
    auto line = stdstring{};
    auto format_version = -1;
    if (stdgetline(wtfile, line)) {
        auto iss = stdstringstream{line};
         First line is the file format version id
        iss  format_version;
        if (iss.fail()  format_version != FORMAT_VERSION) {
            myprintf(Weights file is the wrong version.n);
            return {0, 0};
        } else {
            assert(format_version == FORMAT_VERSION);
            return load_v1_network(wtfile);
        }
    }

    return {0, 0};
}

void Networkinitialize() {
     Prepare rotation table
    for(auto s = 0; s  8; s++) {
        for(auto v = 0; v  BOARD_SQUARES; v++) {
            rotate_nn_idx_table[s][v] = rotate_nn_idx(v, s);
        }
    }

     Load network from file
    size_t channels, residual_blocks;
    stdtie(channels, residual_blocks) = load_network_file(cfg_weightsfile);
    if (channels == 0) {
        exit(EXIT_FAILURE);
    }

    auto weight_index = size_t{0};
     Input convolution
     Winograd transform convolution weights
    conv_weights[weight_index] =
        winograd_transform_f(conv_weights[weight_index],
                             channels, INPUT_CHANNELS);
    weight_index++;

     Residual block convolutions
    for (auto i = size_t{0}; i  residual_blocks  2; i++) {
        conv_weights[weight_index] =
            winograd_transform_f(conv_weights[weight_index],
                                 channels, channels);
        weight_index++;
    }

     Biases are not calculated and are typically zero but some networks might
     still have non-zero biases.
     Move biases to batchnorm means to make the output match without having
     to separately add the biases.
    for (auto i = size_t{0}; i  conv_biases.size(); i++) {
        for (auto j = size_t{0}; j  batchnorm_means[i].size(); j++) {
            batchnorm_means[i][j] -= conv_biases[i][j];
            conv_biases[i][j] = 0.0f;
        }
    }

    for (auto i = size_t{0}; i  bn_val_w1.size(); i++) {
        bn_val_w1[i] -= conv_val_b[i];
        conv_val_b[i] = 0.0f;
    }

    for (auto i = size_t{0}; i  bn_pol_w1.size(); i++) {
        bn_pol_w1[i] -= conv_pol_b[i];
        conv_pol_b[i] = 0.0f;
    }

#ifdef USE_OPENCL
    myprintf(Initializing OpenCL.n);
    opencl.initialize(channels);

    for(const auto & opencl_net  opencl.get_networks()) {
        const auto tuners = opencl_net-getOpenCL().get_sgemm_tuners();

        const auto mwg = tuners[0];
        const auto kwg = tuners[2];
        const auto vwm = tuners[3];

        weight_index = 0;

        const auto m_ceil = ceilMultiple(ceilMultiple(channels, mwg), vwm);
        const auto k_ceil = ceilMultiple(ceilMultiple(INPUT_CHANNELS, kwg), vwm);

        const auto Upad = zeropad_U(conv_weights[weight_index],
                                    channels, INPUT_CHANNELS,
                                    m_ceil, k_ceil);

         Winograd filter transformation changes filter size to 4x4
        opencl_net-push_input_convolution(WINOGRAD_ALPHA, INPUT_CHANNELS,
            channels, Upad,
            batchnorm_means[weight_index], batchnorm_stddivs[weight_index]);
        weight_index++;

         residual blocks
        for (auto i = size_t{0}; i  residual_blocks; i++) {
            const auto Upad1 = zeropad_U(conv_weights[weight_index],
                                         channels, channels,
                                         m_ceil, m_ceil);
            const auto Upad2 = zeropad_U(conv_weights[weight_index + 1],
                                         channels, channels,
                                         m_ceil, m_ceil);
            opencl_net-push_residual(WINOGRAD_ALPHA, channels, channels,
                                      Upad1,
                                      batchnorm_means[weight_index],
                                      batchnorm_stddivs[weight_index],
                                      Upad2,
                                      batchnorm_means[weight_index + 1],
                                      batchnorm_stddivs[weight_index + 1]);
            weight_index += 2;
        }

         Output head convolutions
        opencl_net-push_convolve1(channels, OUTPUTS_POLICY, conv_pol_w);
        opencl_net-push_convolve1(channels, OUTPUTS_VALUE, conv_val_w);
    }
#endif
#ifdef USE_BLAS
#ifndef __APPLE__
#ifdef USE_OPENBLAS
    openblas_set_num_threads(1);
    myprintf(BLAS Core %sn, openblas_get_corename());
#endif
#ifdef USE_MKL
    mkl_set_threading_layer(MKL_THREADING_SEQUENTIAL);
    mkl_set_num_threads(1);
    MKLVersion Version;
    mkl_get_version(&Version);
    myprintf(BLAS core MKL %sn, Version.Processor);
#endif
#endif
#endif
}

#ifdef USE_BLAS
void Networkwinograd_transform_in(const stdvectorfloat& in,
                                    stdvectorfloat& V,
                                    const int C) {
    constexpr auto W = BOARD_SIZE;
    constexpr auto H = BOARD_SIZE;
    constexpr auto WTILES = (W + 1)  2;
    constexpr auto P = WTILES  WTILES;

    stdarraystdarrayfloat, WTILES  2 + 2, WTILES  2 + 2 in_pad;
    for (auto xin = size_t{0}; xin  in_pad.size(); xin++) {
        in_pad[0][xin]     = 0.0f;
        in_pad[H + 1][xin] = 0.0f;
        in_pad[H + 2][xin] = 0.0f;
    }
    for (auto yin = size_t{1}; yin  in_pad[0].size() - 2; yin++) {
        in_pad[yin][0]     = 0.0f;
        in_pad[yin][W + 1] = 0.0f;
        in_pad[yin][W + 2] = 0.0f;
    }

    for (auto ch = 0; ch  C; ch++) {
        for (auto yin = 0; yin  H; yin++) {
            for (auto xin = 0; xin  W; xin++) {
                in_pad[yin + 1][xin + 1] = in[ch(WH) + yinW + xin];
            }
        }
        for (auto block_y = 0; block_y  WTILES; block_y++) {
             Tiles overlap by 2
            const auto yin = 2  block_y;
            for (auto block_x = 0; block_x  WTILES; block_x++) {
                const auto xin = 2  block_x;

                 Calculates transpose(B).x.B
                 B = [[ 1.0,  0.0,  0.0,  0.0],
                      [ 0.0,  1.0, -1.0,  1.0],
                      [-1.0,  1.0,  1.0,  0.0],
                      [ 0.0,  0.0,  0.0, -1.0]]

                using WinogradTile =
                    stdarraystdarrayfloat, WINOGRAD_ALPHA, WINOGRAD_ALPHA;
                WinogradTile T1, T2;

                T1[0][0] = in_pad[yin + 0][xin + 0] - in_pad[yin + 2][xin + 0];
                T1[0][1] = in_pad[yin + 0][xin + 1] - in_pad[yin + 2][xin + 1];
                T1[0][2] = in_pad[yin + 0][xin + 2] - in_pad[yin + 2][xin + 2];
                T1[0][3] = in_pad[yin + 0][xin + 3] - in_pad[yin + 2][xin + 3];
                T1[1][0] = in_pad[yin + 1][xin + 0] + in_pad[yin + 2][xin + 0];
                T1[1][1] = in_pad[yin + 1][xin + 1] + in_pad[yin + 2][xin + 1];
                T1[1][2] = in_pad[yin + 1][xin + 2] + in_pad[yin + 2][xin + 2];
                T1[1][3] = in_pad[yin + 1][xin + 3] + in_pad[yin + 2][xin + 3];
                T1[2][0] = in_pad[yin + 2][xin + 0] - in_pad[yin + 1][xin + 0];
                T1[2][1] = in_pad[yin + 2][xin + 1] - in_pad[yin + 1][xin + 1];
                T1[2][2] = in_pad[yin + 2][xin + 2] - in_pad[yin + 1][xin + 2];
                T1[2][3] = in_pad[yin + 2][xin + 3] - in_pad[yin + 1][xin + 3];
                T1[3][0] = in_pad[yin + 1][xin + 0] - in_pad[yin + 3][xin + 0];
                T1[3][1] = in_pad[yin + 1][xin + 1] - in_pad[yin + 3][xin + 1];
                T1[3][2] = in_pad[yin + 1][xin + 2] - in_pad[yin + 3][xin + 2];
                T1[3][3] = in_pad[yin + 1][xin + 3] - in_pad[yin + 3][xin + 3];

                T2[0][0] = T1[0][0] - T1[0][2];
                T2[0][1] = T1[0][1] + T1[0][2];
                T2[0][2] = T1[0][2] - T1[0][1];
                T2[0][3] = T1[0][1] - T1[0][3];
                T2[1][0] = T1[1][0] - T1[1][2];
                T2[1][1] = T1[1][1] + T1[1][2];
                T2[1][2] = T1[1][2] - T1[1][1];
                T2[1][3] = T1[1][1] - T1[1][3];
                T2[2][0] = T1[2][0] - T1[2][2];
                T2[2][1] = T1[2][1] + T1[2][2];
                T2[2][2] = T1[2][2] - T1[2][1];
                T2[2][3] = T1[2][1] - T1[2][3];
                T2[3][0] = T1[3][0] - T1[3][2];
                T2[3][1] = T1[3][1] + T1[3][2];
                T2[3][2] = T1[3][2] - T1[3][1];
                T2[3][3] = T1[3][1] - T1[3][3];

                const auto offset = ch  P + block_y  WTILES + block_x;
                for (auto i = 0; i  WINOGRAD_ALPHA; i++) {
                    for (auto j = 0; j  WINOGRAD_ALPHA; j++) {
                        V[(iWINOGRAD_ALPHA + j)CP + offset] = T2[i][j];
                    }
                }
            }
        }
    }
}

void Networkwinograd_sgemm(const stdvectorfloat& U,
                             const stdvectorfloat& V,
                             stdvectorfloat& M,
                             const int C, const int K) {
    constexpr auto P = (BOARD_SIZE + 1)  (BOARD_SIZE + 1)  WINOGRAD_ALPHA;

    for (auto b = 0; b  WINOGRAD_TILE; b++) {
        const auto offset_u = b  K  C;
        const auto offset_v = b  C  P;
        const auto offset_m = b  K  P;

        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    K, P, C,
                    1.0f,
                    &U[offset_u], K,
                    &V[offset_v], P,
                    0.0f,
                    &M[offset_m], P);
    }
}

void Networkwinograd_transform_out(const stdvectorfloat& M,
                                     stdvectorfloat& Y,
                                     const int K) {
    constexpr auto W = BOARD_SIZE;
    constexpr auto H = BOARD_SIZE;
    constexpr auto WTILES = (W + 1)  2;
    constexpr auto P = WTILES  WTILES;

    for (auto k = 0; k  K; k++) {
        const auto kHW = k  W  H;
        for (auto block_x = 0; block_x  WTILES; block_x++) {
            const auto x = 2  block_x;
            for (auto block_y = 0; block_y  WTILES; block_y++) {
                const auto y = 2  block_y;

                const auto b = block_y  WTILES + block_x;
                using WinogradTile =
                    stdarraystdarrayfloat, WINOGRAD_ALPHA, WINOGRAD_ALPHA;
                WinogradTile temp_m;
                for (auto xi = 0; xi  WINOGRAD_ALPHA; xi++) {
                    for (auto nu = 0; nu  WINOGRAD_ALPHA; nu++) {
                        temp_m[xi][nu] =
                            M[xi(WINOGRAD_ALPHAKP) + nu(KP)+ kP + b];
                    }
                }

                 Calculates transpose(A).temp_m.A
                    A = [1.0,  0.0],
                        [1.0,  1.0],
                        [1.0, -1.0],
                        [0.0, -1.0]]

                const stdarraystdarrayfloat, 2, 2 o = {
                    temp_m[0][0] + temp_m[0][1] + temp_m[0][2] +
                    temp_m[1][0] + temp_m[1][1] + temp_m[1][2] +
                    temp_m[2][0] + temp_m[2][1] + temp_m[2][2],
                    temp_m[0][1] - temp_m[0][2] - temp_m[0][3] +
                    temp_m[1][1] - temp_m[1][2] - temp_m[1][3] +
                    temp_m[2][1] - temp_m[2][2] - temp_m[2][3],
                    temp_m[1][0] + temp_m[1][1] + temp_m[1][2] -
                    temp_m[2][0] - temp_m[2][1] - temp_m[2][2] -
                    temp_m[3][0] - temp_m[3][1] - temp_m[3][2],
                    temp_m[1][1] - temp_m[1][2] - temp_m[1][3] -
                    temp_m[2][1] + temp_m[2][2] + temp_m[2][3] -
                    temp_m[3][1] + temp_m[3][2] + temp_m[3][3]
                };

                const auto y_ind = kHW + (y)W + (x);
                Y[y_ind] = o[0][0];
                if (x + 1  W) {
                    Y[y_ind + 1] = o[0][1];
                }
                if (y + 1  H) {
                    Y[y_ind + W] = o[1][0];
                    if (x + 1  W) {
                        Y[y_ind + W + 1] = o[1][1];
                    }
                }
            }
        }
    }
}

void Networkwinograd_convolve3(const int outputs,
                                 const stdvectorfloat& input,
                                 const stdvectorfloat& U,
                                 stdvectorfloat& V,
                                 stdvectorfloat& M,
                                 stdvectorfloat& output) {

    constexpr unsigned int filter_len = WINOGRAD_ALPHA  WINOGRAD_ALPHA;
    const auto input_channels = U.size()  (outputs  filter_len);

    winograd_transform_in(input, V, input_channels);
    winograd_sgemm(U, V, M, input_channels, outputs);
    winograd_transform_out(M, output, outputs);
}

templateunsigned int filter_size
void convolve(const size_t outputs,
              const stdvectornet_t& input,
              const stdvectorfloat& weights,
              const stdvectorfloat& biases,
              stdvectorfloat& output) {
     The size of the board is defined at compile time
    constexpr unsigned int width = BOARD_SIZE;
    constexpr unsigned int height = BOARD_SIZE;
    constexpr auto board_squares = width  height;
    constexpr auto filter_len = filter_size  filter_size;
    const auto input_channels = weights.size()  (biases.size()  filter_len);
    const auto filter_dim = filter_len  input_channels;
    assert(outputs  board_squares == output.size());

    stdvectorfloat col(filter_dim  width  height);
    im2colfilter_size(input_channels, input, col);

     Weight shape (output, input, filter_size, filter_size)
     96 18 3 3
     C?aAB + ßC
     outputs[96,19x19] = weights[96,18x3x3] x col[18x3x3,19x19]
     M Number of rows in matrices A and C.
     N Number of columns in matrices B and C.
     K Number of columns in matrix A; number of rows in matrix B.
     lda The size of the first dimention of matrix A; if you are
     passing a matrix A[m][n], the value should be m.
        cblas_sgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B,
                    ldb, beta, C, N);

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                 M        N            K
                outputs, board_squares, filter_dim,
                1.0f, &weights[0], filter_dim,
                &col[0], board_squares,
                0.0f, &output[0], board_squares);

    for (unsigned int o = 0; o  outputs; o++) {
        for (unsigned int b = 0; b  board_squares; b++) {
            output[(o  board_squares) + b] += biases[o];
        }
    }
}

templateunsigned int inputs,
         unsigned int outputs,
         bool ReLU,
         size_t W
stdvectorfloat innerproduct(const stdvectorfloat& input,
                                const stdarrayfloat, W& weights,
                                const stdarrayfloat, outputs& biases) {
    stdvectorfloat output(outputs);

    cblas_sgemv(CblasRowMajor, CblasNoTrans,
                 M     K
                outputs, inputs,
                1.0f, &weights[0], inputs,
                &input[0], 1,
                0.0f, &output[0], 1);

    const auto lambda_ReLU = [](const auto val) { return (val  0.0f) 
                                                          val  0.0f; };
    for (unsigned int o = 0; o  outputs; o++) {
        auto val = biases[o] + output[o];
        if (ReLU) {
            val = lambda_ReLU(val);
        }
        output[o] = val;
    }

    return output;
}

template size_t spatial_size
void batchnorm(const size_t channels,
               stdvectorfloat& data,
               const float const means,
               const float const stddivs,
               const float const eltwise = nullptr)
{
    const auto lambda_ReLU = [](const auto val) { return (val  0.0f) 
                                                          val  0.0f; };
    for (auto c = size_t{0}; c  channels; ++c) {
        const auto mean = means[c];
        const auto scale_stddiv = stddivs[c];

        if (eltwise == nullptr) {
             Classical BN
            const auto arr = &data[c  spatial_size];
            for (auto b = size_t{0}; b  spatial_size; b++) {
                arr[b] = lambda_ReLU(scale_stddiv  (arr[b] - mean));
            }
        } else {
             BN + residual add
            const auto arr = &data[c  spatial_size];
            const auto res = &eltwise[c  spatial_size];
            for (auto b = size_t{0}; b  spatial_size; b++) {
                arr[b] = lambda_ReLU((scale_stddiv  (arr[b] - mean)) + res[b]);
            }
        }
    }
}

void Networkforward_cpu(const stdvectorfloat& input,
                          stdvectorfloat& output_pol,
                          stdvectorfloat& output_val) {
     Input convolution
    constexpr auto width = BOARD_SIZE;
    constexpr auto height = BOARD_SIZE;
    constexpr auto tiles = (width + 1)  (height + 1)  4;
     Calculate output channels
    const auto output_channels = conv_biases[0].size();
     input_channels is the maximum number of input channels of any
     convolution. Residual blocks are identical, but the first convolution
     might be bigger when the network has very few filters
    const auto input_channels = stdmax(static_castsize_t(output_channels),
                                         static_castsize_t(INPUT_CHANNELS));
    auto conv_out = stdvectorfloat(output_channels  width  height);

    auto V = stdvectorfloat(WINOGRAD_TILE  input_channels  tiles);
    auto M = stdvectorfloat(WINOGRAD_TILE  output_channels  tiles);

    winograd_convolve3(output_channels, input, conv_weights[0], V, M, conv_out);
    batchnormBOARD_SQUARES(output_channels, conv_out,
                             batchnorm_means[0].data(),
                             batchnorm_stddivs[0].data());

     Residual tower
    auto conv_in = stdvectorfloat(output_channels  width  height);
    auto res = stdvectorfloat(output_channels  width  height);
    for (auto i = size_t{1}; i  conv_weights.size(); i += 2) {
        auto output_channels = conv_biases[i].size();
        stdswap(conv_out, conv_in);
        winograd_convolve3(output_channels, conv_in,
                           conv_weights[i], V, M, conv_out);
        batchnormBOARD_SQUARES(output_channels, conv_out,
                                 batchnorm_means[i].data(),
                                 batchnorm_stddivs[i].data());

        output_channels = conv_biases[i + 1].size();
        stdswap(conv_in, res);
        stdswap(conv_out, conv_in);
        winograd_convolve3(output_channels, conv_in,
                           conv_weights[i + 1], V, M, conv_out);
        batchnormBOARD_SQUARES(output_channels, conv_out,
                                 batchnorm_means[i + 1].data(),
                                 batchnorm_stddivs[i + 1].data(),
                                 res.data());
    }
    convolve1(OUTPUTS_POLICY, conv_out, conv_pol_w, conv_pol_b, output_pol);
    convolve1(OUTPUTS_VALUE, conv_out, conv_val_w, conv_val_b, output_val);
}

templatetypename T
T relative_difference(const T a, const T b) {
     Handle NaN
    if (stdisnan(a)  stdisnan(b)) {
        return stdnumeric_limitsTmax();
    }

    constexpr auto small_number = 1e-3f;
    auto fa = stdfabs(a);
    auto fb = stdfabs(b);

    if (fa  small_number && fb  small_number) {
         Handle sign difference
        if ((a  0) != (b  0)) {
            return stdnumeric_limitsTmax();
        }
    } else {
         Handle underflow
        fa = stdmax(fa, small_number);
        fb = stdmax(fb, small_number);
    }

    return fabs(fa - fb)  stdmin(fa, fb);
}

void compare_net_outputs(stdvectorfloat& data,
                         stdvectorfloat& ref) {
     We accept an error up to 5%, but output values
     smaller than 11000th are rounded up for the comparison.
    constexpr auto relative_error = 5e-2f;
    for (auto idx = size_t{0}; idx  data.size(); ++idx) {
        const auto err = relative_difference(data[idx], ref[idx]);
        if (err  relative_error) {
            printf(Error in OpenCL calculation expected %f got %f 
                   (error=%f%%)n, ref[idx], data[idx], err  100.0);
            printf(Update your GPU drivers or reduce the amount of games 
                   played simultaneously.n);
            throw stdruntime_error(OpenCL self-check mismatch.);
        }
    }
}
#endif

stdvectorfloat softmax(const stdvectorfloat& input,
                           const float temperature = 1.0f) {
    auto output = stdvectorfloat{};
    output.reserve(input.size());

    const auto alpha = stdmax_element(cbegin(input), cend(input));
    auto denom = 0.0f;

    for (const auto in_val  input) {
        auto val = stdexp((in_val - alpha)  temperature);
        denom += val;
        output.push_back(val);
    }

    for (auto& out  output) {
        out = denom;
    }

    return output;
}

NetworkNetresult Networkget_scored_moves(
    const GameState const state, const Ensemble ensemble,
    const int rotation, const bool skip_cache) {
    Netresult result;
    if (state-board.get_boardsize() != BOARD_SIZE) {
        return result;
    }

    if (!skip_cache) {
         See if we already have this in the cache.
        if (NNCacheget_NNCache().lookup(state-board.get_hash(), result)) {
            return result;
        }
    }

    NNPlanes planes;
    gather_features(state, planes);

    if (ensemble == DIRECT) {
        assert(rotation = 0 && rotation = 7);
        result = get_scored_moves_internal(state, planes, rotation);
    } else {
        assert(ensemble == RANDOM_ROTATION);
        assert(rotation == -1);
        const auto rand_rot = Randomget_Rng().randfix8();
        result = get_scored_moves_internal(state, planes, rand_rot);
    }

     Insert result into cache.
    NNCacheget_NNCache().insert(state-board.get_hash(), result);

    return result;
}

NetworkNetresult Networkget_scored_moves_internal(
    const GameState const state, const NNPlanes & planes, const int rotation) {
    assert(rotation = 0 && rotation = 7);
    assert(INPUT_CHANNELS == planes.size());
    constexpr auto width = BOARD_SIZE;
    constexpr auto height = BOARD_SIZE;
    stdvectornet_t input_data;
    stdvectorfloat policy_data(OUTPUTS_POLICY  width  height);
    stdvectorfloat value_data(OUTPUTS_VALUE  width  height);
     Data layout is input_data[(c  height + h)  width + w]
    input_data.reserve(INPUT_CHANNELS  width  height);
    for (auto c = 0; c  INPUT_CHANNELS; ++c) {
        for (auto h = 0; h  height; ++h) {
            for (auto w = 0; w  width; ++w) {
                const auto rot_idx = rotate_nn_idx_table[rotation][h  width + w];
                input_data.emplace_back(net_t(planes[c][rot_idx]));
            }
        }
    }
#ifdef USE_OPENCL
    opencl.forward(input_data, policy_data, value_data);
#elif defined(USE_BLAS) && !defined(USE_OPENCL)
    forward_cpu(input_data, policy_data, value_data);
#endif
#ifdef USE_OPENCL_SELFCHECK
     Both implementations are available, self-check the OpenCL driver by
     running both with a probability of 12000.
    if (Randomget_Rng().randfixSELFCHECK_PROBABILITY() == 0) {
        auto cpu_policy_data = stdvectorfloat(policy_data.size());
        auto cpu_value_data = stdvectorfloat(value_data.size());
        forward_cpu(input_data, cpu_policy_data, cpu_value_data);
        compare_net_outputs(policy_data, cpu_policy_data);
        compare_net_outputs(value_data, cpu_value_data);
    }
#endif

     Get the moves
    batchnormBOARD_SQUARES(OUTPUTS_POLICY, policy_data,
        bn_pol_w1.data(), bn_pol_w2.data());
    const auto policy_out =
        innerproductOUTPUTS_POLICY  BOARD_SQUARES, BOARD_SQUARES + 1, false(
            policy_data, ip_pol_w, ip_pol_b);
    const auto outputs = softmax(policy_out, cfg_softmax_temp);

     Now get the score
    batchnormBOARD_SQUARES(OUTPUTS_VALUE, value_data,
        bn_val_w1.data(), bn_val_w2.data());
    const auto winrate_data =
        innerproductBOARD_SQUARES, 256, true(value_data, ip1_val_w, ip1_val_b);
    const auto winrate_out =
        innerproduct256, 1, false(winrate_data, ip2_val_w, ip2_val_b);

     Sigmoid
    const auto winrate_sig = (1.0f + stdtanh(winrate_out[0]))  2.0f;

    stdvectorscored_node result;
    for (auto idx = size_t{0}; idx  outputs.size(); idx++) {
        if (idx  BOARD_SQUARES) {
            const auto rot_idx = rotate_nn_idx_table[rotation][idx];
            const auto x = rot_idx % BOARD_SIZE;
            const auto y = rot_idx  BOARD_SIZE;
            const auto rot_vtx = state-board.get_vertex(x, y);
            if (state-board.get_square(rot_vtx) == FastBoardEMPTY) {
                result.emplace_back(outputs[idx], rot_vtx);
            }
        } else {
            result.emplace_back(outputs[idx], FastBoardPASS);
        }
    }

    return stdmake_pair(result, winrate_sig);
}

void Networkshow_heatmap(const FastState const state,
                           const Netresult& result,
                           const bool topmoves) {
    auto moves = result.first;
    stdvectorstdstring display_map;
    stdstring line;

    for (unsigned int y = 0; y  BOARD_SIZE; y++) {
        for (unsigned int x = 0; x  BOARD_SIZE; x++) {
            const auto vtx = state-board.get_vertex(x, y);

            const auto item = stdfind_if(moves.cbegin(), moves.cend(),
                [&vtx](scored_node const& test_item) {
                return test_item.second == vtx;
            });

            auto score = 0;
             Non-empty squares won't be scored
            if (item != moves.cend()) {
                score = int(item-first  1000);
                assert(vtx == item-second);
            }

            line += booststr(boostformat(%3d ) % score);
        }

        display_map.push_back(line);
        line.clear();
    }

    for (int i = display_map.size() - 1; i = 0; --i) {
        myprintf(%sn, display_map[i].c_str());
    }
    assert(result.first.back().second == FastBoardPASS);
    const auto pass_score = int(result.first.back().first  1000);
    myprintf(pass %dn, pass_score);
    myprintf(winrate %fn, result.second);

    if (topmoves) {
        stdstable_sort(rbegin(moves), rend(moves));

        auto cum = 0.0f;
        size_t tried = 0;
        while (cum  0.85f && tried  moves.size()) {
            if (moves[tried].first  0.01f) break;
            myprintf(%1.3f (%s)n,
                    moves[tried].first,
                    state-board.move_to_text(moves[tried].second).c_str());
            cum += moves[tried].first;
            tried++;
        }
    }
}

void Networkfill_input_plane_pair(const FullBoard& board,
                                    BoardPlane& black, BoardPlane& white) {
    auto idx = 0;
    for (auto j = 0; j  BOARD_SIZE; j++) {
        for(auto i = 0; i  BOARD_SIZE; i++) {
            const auto vtx = board.get_vertex(i, j);
            const auto color = board.get_square(vtx);
            if (color != FastBoardEMPTY) {
                if (color == FastBoardBLACK) {
                    black[idx] = true;
                } else {
                    white[idx] = true;
                }
            }
            idx++;
        }
    }
}

void Networkgather_features(const GameState const state, NNPlanes & planes) {
    planes.resize(INPUT_CHANNELS);
    auto& black_to_move = planes[2  INPUT_MOVES];
    auto& white_to_move = planes[2  INPUT_MOVES + 1];

    const auto to_move = state-get_to_move();
    const auto blacks_move = to_move == FastBoardBLACK;

    const auto black_offset = blacks_move  0  INPUT_MOVES;
    const auto white_offset = blacks_move  INPUT_MOVES  0;

    if (blacks_move) {
        black_to_move.set();
    } else {
        white_to_move.set();
    }

    const auto moves = stdminsize_t(state-get_movenum() + 1, INPUT_MOVES);
     Go back in time, fill history boards
    for (auto h = size_t{0}; h  moves; h++) {
         collect white, black occupation planes
        fill_input_plane_pair(state-get_past_board(h),
                              planes[black_offset + h],
                              planes[white_offset + h]);
    }
}

int Networkrotate_nn_idx(const int vertex, int symmetry) {
    assert(vertex = 0 && vertex  BOARD_SQUARES);
    assert(symmetry = 0 && symmetry  8);
    auto x = vertex % BOARD_SIZE;
    auto y = vertex  BOARD_SIZE;
    int newx;
    int newy;

    if (symmetry = 4) {
        stdswap(x, y);
        symmetry -= 4;
    }

    if (symmetry == 0) {
        newx = x;
        newy = y;
    } else if (symmetry == 1) {
        newx = x;
        newy = BOARD_SIZE - y - 1;
    } else if (symmetry == 2) {
        newx = BOARD_SIZE - x - 1;
        newy = y;
    } else {
        assert(symmetry == 3);
        newx = BOARD_SIZE - x - 1;
        newy = BOARD_SIZE - y - 1;
    }

    const auto newvtx = (newy  BOARD_SIZE) + newx;
    assert(newvtx = 0 && newvtx  BOARD_SQUARES);
    return newvtx;
}

