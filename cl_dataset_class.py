import time
import h5py

import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
import torch


###################################################################################################
##################################     AUDIOSET      ##############################################
###################################################################################################

MID2ID_Audioset = {'/g/11b630rrvh': 0, '/g/122z_qxw': 1, '/m/011k_j': 2, '/m/01280g': 3, '/m/012f08': 4, '/m/012n7d': 5, '/m/012ndj': 6, '/m/012xff': 7, '/m/0130jx': 8, '/m/013y1f': 9, '/m/0140xf': 10, '/m/0145m': 11, '/m/014yck': 12, '/m/014zdl': 13, '/m/0150b9': 14, '/m/0155w': 15, '/m/015jpf': 16, '/m/015lz1': 17, '/m/015p6': 18, '/m/015vgc': 19, '/m/015y_n': 20, '/m/0160x5': 21, '/m/0164x2': 22, '/m/016622': 23, '/m/016cjb': 24, '/m/0174k2': 25, '/m/018j2': 26, '/m/018p4k': 27, '/m/018vs': 28, '/m/018w8': 29, '/m/0192l': 30, '/m/0193bn': 31, '/m/0195fx': 32, '/m/0199g': 33, '/m/019jd': 34, '/m/01b82r': 35, '/m/01b9nn': 36, '/m/01b_21': 37, '/m/01bjv': 38, '/m/01bns_': 39, '/m/01c194': 40, '/m/01d380': 41, '/m/01d3sd': 42, '/m/01dwxx': 43, '/m/01g50p': 44, '/m/01g90h': 45, '/m/01glhc': 46, '/m/01h3n': 47, '/m/01h82_': 48, '/m/01h8n0': 49, '/m/01hgjl': 50, '/m/01hhp3': 51, '/m/01hnzm': 52, '/m/01hsr_': 53, '/m/01j2bj': 54, '/m/01j3j8': 55, '/m/01j3sz': 56, '/m/01j423': 57, '/m/01j4z9': 58, '/m/01jg02': 59, '/m/01jg1z': 60, '/m/01jnbd': 61, '/m/01jt3m': 62, '/m/01jwx6': 63, '/m/01kcd': 64, '/m/01lsmm': 65, '/m/01lynh': 66, '/m/01lyv': 67, '/m/01m2v': 68, '/m/01m4t': 69, '/m/01p970': 70, '/m/01qbl': 71, '/m/01rd7k': 72, '/m/01s0ps': 73, '/m/01s0vc': 74, '/m/01sb50': 75, '/m/01sm1g': 76, '/m/01swy6': 77, '/m/01v1d8': 78, '/m/01v_m0': 79, '/m/01w250': 80, '/m/01wy6': 81, '/m/01x3z': 82, '/m/01xq0k1': 83, '/m/01xqw': 84, '/m/01y3hg': 85, '/m/01yg9g': 86, '/m/01yrx': 87, '/m/01z47d': 88, '/m/01z5f': 89, '/m/01z7dr': 90, '/m/02021': 91, '/m/020bb7': 92, '/m/0239kh': 93, '/m/023pjk': 94, '/m/023vsd': 95, '/m/02417f': 96, '/m/0242l': 97, '/m/024dl': 98, '/m/025_jnm': 99, '/m/025rv6n': 100, '/m/025td0t': 101, '/m/025wky1': 102, '/m/0261r1': 103, '/m/0269r2s': 104, '/m/026fgl': 105, '/m/026t6': 106, '/m/026z9': 107, '/m/027m70_': 108, '/m/0283d': 109, '/m/0284vy3': 110, '/m/028ght': 111, '/m/028sqc': 112, '/m/028v0c': 113, '/m/02_41': 114, '/m/02_nn': 115, '/m/02bk07': 116, '/m/02bm9n': 117, '/m/02bxd': 118, '/m/02c8p': 119, '/m/02cjck': 120, '/m/02cz_7': 121, '/m/02dgv': 122, '/m/02f9f_': 123, '/m/02fs_r': 124, '/m/02fsn': 125, '/m/02fxyj': 126, '/m/02g901': 127, '/m/02hnl': 128, '/m/02jz0l': 129, '/m/02k_mr': 130, '/m/02l6bg': 131, '/m/02lkt': 132, '/m/02ll1_': 133, '/m/02mfyn': 134, '/m/02mk9': 135, '/m/02mscn': 136, '/m/02p01q': 137, '/m/02p0sh1': 138, '/m/02p3nc': 139, '/m/02pjr4': 140, '/m/02qldy': 141, '/m/02qmj0d': 142, '/m/02rhddq': 143, '/m/02rlv9': 144, '/m/02rr_': 145, '/m/02rtxlg': 146, '/m/02sgy': 147, '/m/02v2lh': 148, '/m/02w4v': 149, '/m/02x8m': 150, '/m/02x984l': 151, '/m/02y_763': 152, '/m/02yds9': 153, '/m/02z32qm': 154, '/m/02zsn': 155, '/m/030rvx': 156, '/m/0316dw': 157, '/m/0319l': 158, '/m/0326g': 159, '/m/032n05': 160, '/m/032s66': 161, '/m/0342h': 162, '/m/034srq': 163, '/m/0395lw': 164, '/m/039jq': 165, '/m/03_d0': 166, '/m/03cczk': 167, '/m/03cl9h': 168, '/m/03dnzn': 169, '/m/03fwl': 170, '/m/03gvt': 171, '/m/03j1ly': 172, '/m/03k3r': 173, '/m/03kmc9': 174, '/m/03l9g': 175, '/m/03lty': 176, '/m/03m5k': 177, '/m/03m9d0z': 178, '/m/03mb9': 179, '/m/03p19w': 180, '/m/03q5_w': 181, '/m/03q5t': 182, '/m/03qc9zr': 183, '/m/03qjg': 184, '/m/03qtq': 185, '/m/03qtwd': 186, '/m/03r5q_': 187, '/m/03t3fj': 188, '/m/03v3yw': 189, '/m/03vt0': 190, '/m/03w41f': 191, '/m/03wvsk': 192, '/m/03wwcy': 193, '/m/03xq_f': 194, '/m/040b_t': 195, '/m/04179zz': 196, '/m/04229': 197, '/m/042v_gx': 198, '/m/0463cq4': 199, '/m/046dlr': 200, '/m/04_sv': 201, '/m/04brg2': 202, '/m/04ctx': 203, '/m/04cvmfc': 204, '/m/04fgwm': 205, '/m/04fq5q': 206, '/m/04gxbd': 207, '/m/04gy_2': 208, '/m/04k94': 209, '/m/04qvtq': 210, '/m/04rlf': 211, '/m/04rmv': 212, '/m/04rzd': 213, '/m/04s8yn': 214, '/m/04szw': 215, '/m/04v5dt': 216, '/m/04wptg': 217, '/m/04zjc': 218, '/m/04zmvq': 219, '/m/05148p4': 220, '/m/053hz1': 221, '/m/056ks2': 222, '/m/056r_1': 223, '/m/05_wcq': 224, '/m/05fw6t': 225, '/m/05kq4': 226, '/m/05lls': 227, '/m/05mxj0q': 228, '/m/05pd6': 229, '/m/05r5c': 230, '/m/05r5wn': 231, '/m/05r6t': 232, '/m/05rj2': 233, '/m/05rwpb': 234, '/m/05tny_': 235, '/m/05w3f': 236, '/m/05x_td': 237, '/m/05zc1': 238, '/m/05zppz': 239, '/m/0641k': 240, '/m/0642b4': 241, '/m/064t9': 242, '/m/068hy': 243, '/m/068zj': 244, '/m/06_fw': 245, '/m/06_y0by': 246, '/m/06bxc': 247, '/m/06by7': 248, '/m/06bz3': 249, '/m/06cqb': 250, '/m/06cyt0': 251, '/m/06d_3': 252, '/m/06h7j': 253, '/m/06hck5': 254, '/m/06hps': 255, '/m/06j64v': 256, '/m/06j6l': 257, '/m/06mb1': 258, '/m/06ncr': 259, '/m/06q74': 260, '/m/06rqw': 261, '/m/06rvn': 262, '/m/06w87': 263, '/m/06wzb': 264, '/m/06xkwv': 265, '/m/073cg4': 266, '/m/074ft': 267, '/m/078jl': 268, '/m/0790c': 269, '/m/079vc8': 270, '/m/07bgp': 271, '/m/07bjf': 272, '/m/07bm98': 273, '/m/07brj': 274, '/m/07c52': 275, '/m/07c6l': 276, '/m/07cx4': 277, '/m/07gql': 278, '/m/07gxw': 279, '/m/07hvw1': 280, '/m/07jdr': 281, '/m/07k1x': 282, '/m/07kc_': 283, '/m/07lnk': 284, '/m/07m2kt': 285, '/m/07mzm6': 286, '/m/07n_g': 287, '/m/07p6fty': 288, '/m/07p6mqd': 289, '/m/07p78v5': 290, '/m/07p7b8y': 291, '/m/07p9k1k': 292, '/m/07p_0gm': 293, '/m/07pb8fc': 294, '/m/07pbtc8': 295, '/m/07pc8l3': 296, '/m/07pc8lb': 297, '/m/07pczhz': 298, '/m/07pdhp0': 299, '/m/07pdjhy': 300, '/m/07pggtn': 301, '/m/07phhsh': 302, '/m/07phxs1': 303, '/m/07pjjrj': 304, '/m/07pjwq1': 305, '/m/07pk7mg': 306, '/m/07pkxdp': 307, '/m/07pl1bw': 308, '/m/07plct2': 309, '/m/07plz5l': 310, '/m/07pn_8q': 311, '/m/07pp8cl': 312, '/m/07pp_mv': 313, '/m/07ppn3j': 314, '/m/07pqc89': 315, '/m/07pqmly': 316, '/m/07pqn27': 317, '/m/07prgkl': 318, '/m/07pt6mm': 319, '/m/07pt_g0': 320, '/m/07ptfmf': 321, '/m/07ptzwd': 322, '/m/07pws3f': 323, '/m/07pxg6y': 324, '/m/07pyf11': 325, '/m/07pyy8b': 326, '/m/07pzfmf': 327, '/m/07q0h5t': 328, '/m/07q0yl5': 329, '/m/07q2z82': 330, '/m/07q34h3': 331, '/m/07q4ntr': 332, '/m/07q5rw0': 333, '/m/07q6cd_': 334, '/m/07q7njn': 335, '/m/07q8f3b': 336, '/m/07q8k13': 337, '/m/07qb_dv': 338, '/m/07qc9xj': 339, '/m/07qcpgn': 340, '/m/07qcx4z': 341, '/m/07qdb04': 342, '/m/07qf0zm': 343, '/m/07qfgpx': 344, '/m/07qfr4h': 345, '/m/07qh7jl': 346, '/m/07qjznl': 347, '/m/07qjznt': 348, '/m/07qlf79': 349, '/m/07qlwh6': 350, '/m/07qmpdm': 351, '/m/07qn4z3': 352, '/m/07qn5dc': 353, '/m/07qnq_y': 354, '/m/07qqyl4': 355, '/m/07qrkrw': 356, '/m/07qs1cx': 357, '/m/07qsvvw': 358, '/m/07qv4k0': 359, '/m/07qv_d5': 360, '/m/07qv_x_': 361, '/m/07qw_06': 362, '/m/07qwdck': 363, '/m/07qwf61': 364, '/m/07qwyj0': 365, '/m/07qyrcz': 366, '/m/07qz6j3': 367, '/m/07r04': 368, '/m/07r10fb': 369, '/m/07r4gkf': 370, '/m/07r4k75': 371, '/m/07r4wb8': 372, '/m/07r5c2p': 373, '/m/07r5v4s': 374, '/m/07r660_': 375, '/m/07r67yg': 376, '/m/07r81j2': 377, '/m/07r_25d': 378, '/m/07r_80w': 379, '/m/07r_k2n': 380, '/m/07rbp7_': 381, '/m/07rc7d9': 382, '/m/07rcgpl': 383, '/m/07rdhzs': 384, '/m/07rgkc5': 385, '/m/07rgt08': 386, '/m/07rjwbb': 387, '/m/07rjzl8': 388, '/m/07rkbfh': 389, '/m/07rknqz': 390, '/m/07rn7sz': 391, '/m/07rpkh9': 392, '/m/07rqsjt': 393, '/m/07rrh0c': 394, '/m/07rrlb6': 395, '/m/07rv4dm': 396, '/m/07rv9rh': 397, '/m/07rwj3x': 398, '/m/07rwm0c': 399, '/m/07ryjzk': 400, '/m/07s02z0': 401, '/m/07s04w4': 402, '/m/07s0dtb': 403, '/m/07s0s5r': 404, '/m/07s12q4': 405, '/m/07s13rg': 406, '/m/07s2xch': 407, '/m/07s34ls': 408, '/m/07s72n': 409, '/m/07s8j8t': 410, '/m/07sbbz2': 411, '/m/07sk0jz': 412, '/m/07sq110': 413, '/m/07sr1lc': 414, '/m/07st88b': 415, '/m/07st89h': 416, '/m/07svc2k': 417, '/m/07swgks': 418, '/m/07sx8x_': 419, '/m/07szfh9': 420, '/m/07xzm': 421, '/m/07y_7': 422, '/m/07yv9': 423, '/m/081rb': 424, '/m/0838f': 425, '/m/083vt': 426, '/m/085jw': 427, '/m/08cyft': 428, '/m/08dckq': 429, '/m/08j51y': 430, '/m/08p9q4': 431, '/m/0912c9': 432, '/m/0939n_': 433, '/m/093_4n': 434, '/m/096m7z': 435, '/m/098_xr': 436, '/m/09b5t': 437, '/m/09ct_': 438, '/m/09d1b1': 439, '/m/09d5_': 440, '/m/09ddx': 441, '/m/09f96': 442, '/m/09hlz4': 443, '/m/09l8g': 444, '/m/09ld4': 445, '/m/09t49': 446, '/m/09x0r': 447, '/m/09xqv': 448, '/m/0_1c': 449, '/m/0_ksk': 450, '/m/0b9m1': 451, '/m/0b_fwt': 452, '/m/0bcdqg': 453, '/m/0bm02': 454, '/m/0bpl036': 455, '/m/0brhx': 456, '/m/0bt9lr': 457, '/m/0btp2': 458, '/m/0bzvm2': 459, '/m/0c1dj': 460, '/m/0c1tlg': 461, '/m/0c2wf': 462, '/m/0c3f7m': 463, '/m/0cdnk': 464, '/m/0cfdd': 465, '/m/0ch8v': 466, '/m/0chx_': 467, '/m/0cj0r': 468, '/m/0cmf2': 469, '/m/0d31p': 470, '/m/0d4wf': 471, '/m/0d8_n': 472, '/m/0dbvp': 473, '/m/0dgbq': 474, '/m/0dgw9r': 475, '/m/0dl5d': 476, '/m/0dl83': 477, '/m/0dl9sf8': 478, '/m/0dls3': 479, '/m/0dq0md': 480, '/m/0dv3j': 481, '/m/0dv5r': 482, '/m/0dwsp': 483, '/m/0dwt5': 484, '/m/0dwtp': 485, '/m/0dxrf': 486, '/m/0f8s22': 487, '/m/0fd3y': 488, '/m/0ffhf': 489, '/m/0fjy1': 490, '/m/0fqfqc': 491, '/m/0fw86': 492, '/m/0fx80y': 493, '/m/0fx9l': 494, '/m/0g12c5': 495, '/m/0g293': 496, '/m/0g6b5': 497, '/m/0gg8l': 498, '/m/0ggq0m': 499, '/m/0ggx5q': 500, '/m/0ghcn6': 501, '/m/0glt670': 502, '/m/0gvgw0': 503, '/m/0gy1t2s': 504, '/m/0gywn': 505, '/m/0h0rv': 506, '/m/0h2mp': 507, '/m/0h9mv': 508, '/m/0hdsk': 509, '/m/0hg7b': 510, '/m/0hgq8df': 511, '/m/0hsrw': 512, '/m/0j2kx': 513, '/m/0j45pbj': 514, '/m/0j6m2': 515, '/m/0jb2l': 516, '/m/0jbk': 517, '/m/0jtg0': 518, '/m/0k4j': 519, '/m/0k5j': 520, '/m/0k65p': 521, '/m/0l14_3': 522, '/m/0l14gg': 523, '/m/0l14j_': 524, '/m/0l14jd': 525, '/m/0l14l2': 526, '/m/0l14md': 527, '/m/0l14qv': 528, '/m/0l14t7': 529, '/m/0l156b': 530, '/m/0l156k': 531, '/m/0l15bq': 532, '/m/0l7xg': 533, '/m/0llzx': 534, '/m/0ln16': 535, '/m/0ltv': 536, '/m/0lyf6': 537, '/m/0m0jc': 538, '/m/0mbct': 539, '/m/0md09': 540, '/m/0mkg': 541, '/m/0ngt1': 542, '/m/0xzly': 543, '/m/0y4f8': 544, '/m/0ytgt': 545, '/m/0z9c': 546, '/m/0zmy2j9': 547, '/t/dd00001': 548, '/t/dd00002': 549, '/t/dd00003': 550, '/t/dd00004': 551, '/t/dd00005': 552, '/t/dd00006': 553, '/t/dd00012': 554, '/t/dd00013': 555, '/t/dd00018': 556, '/t/dd00031': 557, '/t/dd00032': 558, '/t/dd00033': 559, '/t/dd00034': 560, '/t/dd00035': 561, '/t/dd00036': 562, '/t/dd00037': 563, '/t/dd00038': 564, '/t/dd00048': 565, '/t/dd00061': 566, '/t/dd00065': 567, '/t/dd00066': 568, '/t/dd00067': 569, '/t/dd00077': 570, '/t/dd00088': 571, '/t/dd00091': 572, '/t/dd00092': 573, '/t/dd00098': 574, '/t/dd00099': 575, '/t/dd00108': 576, '/t/dd00109': 577, '/t/dd00110': 578, '/t/dd00112': 579, '/t/dd00118': 580, '/t/dd00121': 581, '/t/dd00122': 582, '/t/dd00123': 583, '/t/dd00125': 584, '/t/dd00126': 585, '/t/dd00127': 586, '/t/dd00128': 587, '/t/dd00129': 588, '/t/dd00130': 589, '/t/dd00133': 590, '/t/dd00134': 591, '/t/dd00135': 592, '/t/dd00136': 593, '/t/dd00138': 594, '/t/dd00139': 595, '/t/dd00141': 596, '/t/dd00142': 597, '/t/dd00143': 598, '/t/dd00144': 599, '/t/dd00147': 600}
class Audioset(Dataset):
    def __init__(self, data_path, selected_classes=list(MID2ID_Audioset.values()), test=False, debug=False):
        # start_time = time.time()
        if test: 
            self.dataset = h5py.File(f'{data_path}/h5s/audioset_eval.h5', "r")
            # raise Exception("Audioset Evaluation dataset not implemented yet!")
        else:
            self.dataset = h5py.File(f'{data_path}/h5s/audioset_VDS.h5', "r")
        # end_time = time.time()
        # print(f"H5 loading time: {round(end_time-start_time, 2)} s - ")
        self.selected_classes=sorted(selected_classes)
        
        # start_time = time.time()
        
        self.task_indexes = self.__get_task_dataset_indexes_from_hd5__(debug=debug)
        np.random.shuffle(self.task_indexes)
        # end_time = time.time()
        # print(f"Costo trova sample: {round(end_time-start_time, 2)} s - ")

        # start_time = time.time()
        self.data = self.dataset["mel_spectrogram"]
        self.targets = self.dataset["one_hot_labels"][self.task_indexes]
        self.targets = self.targets[:, self.selected_classes]
        self.data = self.data[self.task_indexes]
        # end_time = time.time()
        # print(f"Costo trova sample da task indexes: {round(end_time-start_time, 2)} s - ")

        self.start = 0
        self.end = len(self.task_indexes)

        # self.pos_weight = self.__class_imbalance_weights__()

    def __len__(self):
        return self.end-self.start
    
    def __getitem__(self, index):
        # start_time = time.time()
        d, l = torch.from_numpy(self.data[self.start+index]).t().unsqueeze(0), torch.from_numpy(self.targets[self.start+index].astype(np.float64))
        # end_time = time.time()
        # print(f"Costo __getitem__: {round(end_time-start_time, 2)} s - ")
        return d, l
    
    def __class_imbalance_weights__(self):
        N = self.__len__()
        labels = np.stack([ self.__getitem__(i)[1] for i in range(N) ])  
        pos_counts = labels.sum(axis=0)                                
        neg_counts = N - pos_counts                                     

        pos_counts = np.where(pos_counts == 0, 1, pos_counts)
        pos_weight = neg_counts / pos_counts                            
        return pos_weight

    
    def __get_task_dataset_indexes_from_hd5__(self, debug):

        data = []
        if type(self.dataset) is not list:
            
            if debug:
                length = 10000
            else:
                length = len(self.dataset["filenames"])
            
            labels = self.dataset["one_hot_labels"]
            res = np.any(labels[:length, self.selected_classes], axis=1)
            data = np.nonzero(res)[0].astype(np.int64)
        
        else:
            
            for i, entry in enumerate(self.dataset):
                res = np.any(entry["one_hot_label"][:][self.selected_classes])
                if res: 
                    data.append(i)

        return data    

    def __del__(self):
        try:
            if hasattr(self, "dataset") and self.dataset:
                self.dataset.close()
        except Exception as e:
                pass
        
        

###################################################################################################
##################################     FSD50K      ##############################################
###################################################################################################

MID2ID_Fsd50k = {"/m/07q2z82": 0, "/m/0mkg": 1, "/m/042v_gx": 2, "/m/0k5j": 3, "/m/07pp_mv": 4, "/m/0jbk": 5, "/m/028ght": 6, "/m/05tny_": 7, "/m/0bm02": 8, "/m/018vs": 9, "/m/03dnzn": 10, "/m/0395lw": 11, "/m/0199g": 12, "/m/0gy1t2s": 13, "/m/015p6": 14, "/m/020bb7": 15, "/m/019jd": 16, "/m/0dv3j": 17, "/m/07qqyl4": 18, "/m/0l14_3": 19, "/m/01kcd": 20, "/m/0lyf6": 21, "/m/03q5_w": 22, "/m/01bjv": 23, "/m/07pjwq1": 24, "/m/0dv5r": 25, "/m/0k4j": 26, "/t/dd00134": 27, "/m/01yrx": 28, "/m/07rkbfh": 29, "/m/053hz1": 30, "/m/03cczk": 31, "/m/09b5t": 32, "/m/0ytgt": 33, "/m/0f8s22": 34, "/m/07q7njn": 35, "/m/07pggtn": 36, "/m/07rgt08": 37, "/m/03w41f": 38, "/m/0l15bq": 39, "/m/01x3z": 40, "/m/0242l": 41, "/m/01m2v": 42, "/m/01h8n0": 43, "/m/01b_21": 44, "/m/0239kh": 45, "/m/07qs1cx": 46, "/m/07pzfmf": 47, "/m/0bm0k": 48, "/m/09xqv": 49, "/m/04s8yn": 50, "/m/03qtwd": 51, "/t/dd00112": 52, "/m/07plct2": 53, "/m/0463cq4": 54, "/m/0642b4": 55, "/m/023pjk": 56, "/m/01qbl": 57, "/m/04brg2": 58, "/m/0bt9lr": 59, "/m/068hy": 60, "/t/dd00071": 61, "/m/02dgv": 62, "/m/03wwcy": 63, "/m/0fqfqc": 64, "/m/01d380": 65, "/m/07r5v4s": 66, "/m/026t6": 67, "/m/02hnl": 68, "/m/02sgy": 69, "/m/02mk9": 70, "/t/dd00130": 71, "/m/014zdl": 72, "/m/02_nn": 73, "/t/dd00004": 74, "/m/02zsn": 75, "/m/07p7b8y": 76, "/m/025_jnm": 77, "/m/02_41": 78, "/m/0g6b5": 79, "/m/0cmf2": 80, "/m/025rv6n": 81, "/m/09ld4": 82, "/m/0dxrf": 83, "/m/07s0dtb": 84, "/m/07r660_": 85, "/m/039jq": 86, "/m/0dwtp": 87, "/m/0mbct": 88, "/m/0ghcn6": 89, "/m/0342h": 90, "/m/01dwxx": 91, "/m/032s66": 92, "/m/07swgks": 93, "/m/03l9g": 94, "/m/0k65p": 95, "/m/03qjg": 96, "/m/03m5k": 97, "/m/03qtq": 98, "/m/07rjwbb": 99, "/t/dd00012": 100, "/m/09l8g": 101, "/m/07pb8fc": 102, "/m/03vt0": 103, "/m/05148p4": 104, "/m/03v3yw": 105, "/m/07r4wb8": 106, "/m/01j3sz": 107, "/m/04k94": 108, "/m/0ch8v": 109, "/t/dd00003": 110, "/m/05zppz": 111, "/m/0j45pbj": 112, "/m/0dwsp": 113, "/m/02x984l": 114, "/t/dd00077": 115, "/m/07qrkrw": 116, "/m/0fx9l": 117, "/m/012f08": 118, "/m/04_sv": 119, "/m/04rlf": 120, "/m/04szw": 121, "/m/05kq4": 122, "/m/013y1f": 123, "/m/05mxj0q": 124, "/m/0l14md": 125, "/m/05r5c": 126, "/m/0fx80y": 127, "/m/07prgkl": 128, "/m/0_ksk": 129, "/m/01m4t": 130, "/m/02yds9": 131, "/m/0ltv": 132, "/m/06d_3": 133, "/m/06mb1": 134, "/m/07r10fb": 135, "/m/02bm9n": 136, "/m/07qn4z3": 137, "/m/05r5wn": 138, "/m/09hlz4": 139, "/m/01hnzm": 140, "/m/06h7j": 141, "/m/01b82r": 142, "/m/01lsmm": 143, "/m/01hgjl": 144, "/m/03qc9zr": 145, "/m/07q8k13": 146, "/m/07rn7sz": 147, "/m/07p6fty": 148, "/m/07plz5l": 149, "/m/015lz1": 150, "/m/0130jx": 151, "/m/03kmc9": 152, "/m/06_fw": 153, "/m/07rjzl8": 154, "/m/02y_763": 155, "/m/06rvn": 156, "/m/01hsr_": 157, "/m/09x0r": 158, "/m/0brhx": 159, "/m/07rrlb6": 160, "/m/07q6cd_": 161, "/m/0j6m2": 162, "/m/07s0s5r": 163, "/m/0195fx": 164, "/m/01p970": 165, "/m/07brj": 166, "/m/07qcpgn": 167, "/m/07qcx4z": 168, "/m/07cx4": 169, "/m/07qnq_y": 170, "/m/0ngt1": 171, "/m/0jb2l": 172, "/m/07qjznt": 173, "/m/07qjznl": 174, "/m/01jt3m": 175, "/m/07k1x": 176, "/m/0btp2": 177, "/m/07jdr": 178, "/m/07pqc89": 179, "/m/07r04": 180, "/m/07gql": 181, "/m/0c2wf": 182, "/m/0316dw": 183, "/m/07yv9": 184, "/m/0912c9": 185, "/m/07pbtc8": 186, "/m/0838f": 187, "/m/02jz0l": 188, "/m/034srq": 189, "/m/02rtxlg": 190, "/m/07rqsjt": 191, "/m/01280g": 192, "/m/03m9d0z": 193, "/m/026fgl": 194, "/m/085jw": 195, "/m/083vt": 196, "/m/081rb": 197, "/m/07sr1lc": 198, "/m/01s0vc": 199}
class Fsd50k(Dataset):
    def __init__(self, data_path, selected_classes=list(MID2ID_Fsd50k.values()), test=False, debug=False):
        if test: 
            self.dataset = h5py.File(f'{data_path}/h5s/fsd50k_eval.h5', "r")
        else:
            self.dataset = h5py.File(f'{data_path}/h5s/fsd50k_dev.h5', "r")

        self.selected_classes=sorted(selected_classes)
        # self.task_indexes = self.__get_task_dataset_indexes_from_hd5__(debug)
        np.random.shuffle(self.task_indexes)
        if debug:
            self.task_indexes = self.task_indexes[:round(len(self.task_indexes)*0.1)]

        self.data = self.dataset["mel_spectrogram"]
        self.targets = self.dataset["one_hot_labels"]
        
        
        self.start = 0
        self.end = len(self.task_indexes)

        # self.pos_weight = self.__class_imbalance_weights__()

    def __len__(self):
        return self.end-self.start
    
    def __getitem__(self, index):
        return torch.from_numpy(self.data[self.start+index]).t().unsqueeze(0), torch.from_numpy(self.targets[self.start+index][self.selected_classes].astype(np.float64))

        # return torch.from_numpy(self.data[self.task_indexes[self.start+index]]).t().unsqueeze(0), torch.from_numpy(self.targets[self.task_indexes[self.start+index]][self.selected_classes].astype(np.float64))
    
    def __class_imbalance_weights__(self):
        N = self.__len__()
        labels = np.stack([ self.__getitem__(i)[1] for i in range(N) ])  
        pos_counts = labels.sum(axis=0)                                
        neg_counts = N - pos_counts                                     

        pos_counts = np.where(pos_counts == 0, 1, pos_counts)
        pos_weight = neg_counts / pos_counts                            
        return pos_weight

    
    def __get_task_dataset_indexes_from_hd5__(self, debug):

        data = []
        if type(self.dataset) is not list:
            if debug:
                length = 10000
            else:
                length = len(self.dataset["filenames"])
            for i in range(length):
                entry = self.dataset["one_hot_labels"][i]
                res = np.any(entry[self.selected_classes])
                if res: 
                    data.append(i)
        
        else:
            for i, entry in enumerate(self.dataset):
                res = np.any(entry["one_hot_label"][:][self.selected_classes])
                if res: 
                    data.append(i)

        return data    

    def __del__(self):
        try:
            if hasattr(self, "dataset") and self.dataset:
                self.dataset.close()
        except Exception as e:
                pass
        
        