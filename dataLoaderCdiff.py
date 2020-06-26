import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import copy


class cdiffDataLoader():
    def __init__(self, file_name="CDiffMetabolomics.xlsx", file_name16s='dada2-seqtab-nochim.xlsx'):
        self.filename = file_name
    
        self.xl = pd.ExcelFile(self.filename)

        self.cdiff_raw = self.xl.parse('OrigScale', header=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], index_col=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
        # self.cdiff_raw.columns = self.cdiff_raw.columns.str.replace("Carrier", "Cleared")
        header_lst = [i[4:]
                      for i in self.cdiff_raw.columns.values if 'Unnamed' not in i[0]]
        self.header_labels = header_lst[0]

        # get rid of repeat values (1a, 1b, 1c, etc.)
        tmpts = [c[4].split('-')[-1] for c in self.cdiff_raw.columns.values]
        ix = []
        for i, tmpt in enumerate(tmpts):
            try:
                float(tmpt)
            except:
                ix.append(i)
        self.cdiff_raw = self.cdiff_raw.iloc[:, [
            j for j, c in enumerate(self.cdiff_raw.columns) if j not in ix]]
        
        header_lst = [i[4:] for i in self.cdiff_raw.columns.values if 'Unnamed' not in i[0]]
        self.header_names = header_lst
        # pt_info_dict = {header_names[j][0]:{header_labels[i]:header_names[j][i] for i in range(1,len(header_labels))} for j in range(len(header_names))}

        self.row_names = self.cdiff_raw.index.values[1:]
        self.row_labels = self.cdiff_raw.index.values[0]
        self.metabolome_info_dict = {self.row_names[j][1]: {self.row_labels[i]: self.row_names[j][i] for i in range(len(self.row_labels))} for j in range(len(self.row_names))}

        self.cdiff_raw = self.cdiff_raw.iloc[1:, :]

        colnames = [x[4] for x in self.cdiff_raw.columns.values]
        self.cdiff_data = pd.DataFrame(np.array(self.cdiff_raw), columns=colnames)

        self.raw16s = pd.ExcelFile(file_name16s)
        self.raw16s = self.raw16s.parse()

        self.labels16s_dict = {101:0,102:0,103:0,105:1,106:0,108:1,109:0,114:0,115:0,116:0,119:1,120:1,121:1,123:0,124:1,\
            126:0,127:0,129:0,130:1,131:0,132:0,133:0,134:1,136:1,138:0,139:0,140:0,141:1,142:0,143:1,144:0,145:0,146:1,147:0,148:1,\
                149:1,150:0,151:0,152:0,153:0,155:0,156:1,158:1,160:0,161:1,162:1,163:1,165:1,167:1,168:'g',169:'g',170:'g',171:'g',\
                    173:1,174:'g',175:'g',107:'b',117:'b',118:'b',164:'b'}

        self.cl_dict = {0:'Cleared',1:'Recur','g':'Unknown','b':'Unknown'}
        dcols = ['-'.join(x.split('-')[1:])
                 for x in self.raw16s.columns.values[1:]]
        dcol = []
        for x in dcols:
            if len(x.split('-')[1]) == 2:
                dcol.append('.'.join([x[:5], x[-1]]))
            else:
                dcol.append(x)
        dcols = dcol
        self.data16s = pd.DataFrame(
            np.array(self.raw16s.iloc[:, 1:]), columns=dcols, index=self.raw16s.iloc[:, 0])

        self.data16s = self.make_proportions(self.data16s)

        self.make_pt_dict(self.cdiff_data)
        self.metabolome_pts = np.concatenate(
            [[self.pt_info_dict[i][k]['CLIENT SAMPLE ID'] for k in self.pt_info_dict[i].keys()] for i in self.pt_info_dict.keys()])
        self.microbiome_pts = dcols
        filt_out = self.filter_metabolites(40)
        self.all_16s_info_dict = self.add_16s_to_info_dict(self.data16s, self.pt_info_dict)
        # self.info_dict_16s = self.make_16s_dict(self.data16s)

    def make_pt_dict(self, data, idxs=None):

        pt_names = np.array([h[0].split('-')[0] for h in self.header_names])
        pts = []
        for i,n in enumerate(np.unique(pt_names)):
            pts.extend([str(i+1) + '.' + str(j) for j in range(len(np.where(pt_names == n)[0]))])

        ts = np.array([h[0].split('-')[1] for h in self.header_names])
        self.times = []
        for el in ts:
            self.times.append(float(el))
            # except:
            #     tm = str(ord(list(el)[1])-97)
            #     self.times.append(float(list(el)[0] + '.' + tm))
        idx_num = np.where(np.array(self.row_labels) == 'BIOCHEMICAL')[0][0]
        idx_val = [r[idx_num] for r in self.row_names]
        if idxs is None:
            cdiff_raw_sm = data
            cdiff_raw_sm = cdiff_raw_sm.fillna(0)
            # import pdb
            # pdb.set_trace()
            self.data = pd.DataFrame(
                np.array(data), columns=self.header_names, index=self.row_names)
        else:
            cdiff_raw_sm = pd.DataFrame(np.array(data)[idxs, :], index = np.array(idx_val)[idxs])
            cdiff_raw_sm = cdiff_raw_sm.fillna(0)
            self.data = pd.DataFrame(
                np.array(data)[idxs, :], columns=self.header_names, index=np.array(self.row_names)[idxs])

        self.data_sm = cdiff_raw_sm
        self.pt_info_dict = {}
        pt_key = pt_names[0]
        self.pt_info_dict[pt_key] = {}
        

        for j in range(len(self.header_names)):
            if j != 0 and pt_names[j] != pt_names[j-1]:
                pt_key = pt_names[j]
                self.pt_info_dict[pt_key] = {}
                self.pt_info_dict[pt_key][self.times[j]] = {
                    self.header_labels[i]: self.header_names[j][i] for i in range(len(self.header_labels))}
                self.pt_info_dict[pt_key][self.times[j]].update(
                    {'DATA': cdiff_raw_sm.iloc[:, j]})

            else:
                self.pt_info_dict[pt_key][self.times[j]] = {
                    self.header_labels[i]: self.header_names[j][i] for i in range(len(self.header_labels))}
                self.pt_info_dict[pt_key][self.times[j]].update(
                    {'DATA': cdiff_raw_sm.iloc[:, j]})

    def add_16s_to_info_dict(self, data16s, pt_info_dict):
        pt_info_dict_all16s = copy.deepcopy(pt_info_dict)
        all_pts = np.unique(list(self.cdiff_data)+ list(self.data16s))
        for key in all_pts:
            key_pt = key.split('-')[0]
            key_tmpt = float(key.split('-')[1])
            if key_pt in pt_info_dict.keys() and key_tmpt in pt_info_dict[key_pt].keys():
                pt_label = key
                try:
                    col_16s = data16s[pt_label]
                    pt_info_dict[key_pt][key_tmpt]['16s'] = col_16s
                    pt_info_dict_all16s[key_pt][key_tmpt]['16s'] = col_16s
                except:
                    continue
                    # if key_pt not in pt_info_dict_all16s.keys():
                    #     pt_info_dict_all16s[key_pt] = {}
                    # if key_tmpt not in pt_info_dict_all16s[key_pt].keys():
                    #     pt_info_dict_all16s[key_pt][key_tmpt] = {}
                    # if pt_label in data16s.columns.values:
                    #     col_16s = data16s[pt_label]
                    #     pt_info_dict_all16s[key_pt][key_tmpt]['16s'] = col_16s
            else:
                # if key_pt == '101':
                #     import pdb; pdb.set_trace()
                pt_label = key
                col_16s = data16s[pt_label]
                try:
                    pt_info_dict[key_pt][key_tmpt]['16s'] = col_16s
                    pt_info_dict_all16s[key_pt][key_tmpt]['16s'] = col_16s
                except:
                    if key_pt not in pt_info_dict_all16s.keys():
                        pt_info_dict_all16s[key_pt] = {}
                    if key_tmpt not in pt_info_dict_all16s[key_pt].keys():
                        pt_info_dict_all16s[key_pt][key_tmpt] = {}
                    pt_info_dict_all16s[key_pt][key_tmpt]['16s'] = col_16s
                lab = self.labels16s_dict[int(pt_label.split('-')[0])]
                label = self.cl_dict[lab]
                pt_info_dict_all16s[key_pt][key_tmpt]['PATIENT STATUS (BWH)'] = label

        return pt_info_dict_all16s
    
    def make_proportions(self, data):
        total_counts = np.sum(data, 0)
        data = data / total_counts
        return data

    def index_by(self, colname, rowname):
        cols = self.data.columns.values
        ix1 = np.where(np.array(self.header_labels) == colname)[0][0]

        c_map = {cols[i]: cols[i][ix1] for i in range(len(cols))}
        # colnames = [cols[i][ix1] for i in range(len(cols))]

        rows = self.data.index.values
        ix2 = np.where(np.array(self.row_labels) == rowname)[0][0]

        # rownames = [rows[i][ix2] for i in range(len(rows))]

        r_map = {rows[i]: rows[i][ix2] for i in range(len(rows))}
        data = self.data.rename(columns=c_map, index=r_map)
        return data

    def filter_metabolites(self,val):
        c = []
        for pt in self.pt_info_dict.keys():
            c.append(np.sum(np.vstack([self.pt_info_dict[pt][t]['DATA']
                                    for t in self.pt_info_dict[pt].keys()]), 0))
        cts = np.vstack(c).T
        cts[cts > 0] = 1
        counts = np.sum(cts, 1)

        filt_out = np.where(counts > val)[0]
        self.make_pt_dict(np.array(self.data_sm), idxs=filt_out)
        return filt_out

    def filter_microbes(self, val):
        c = []
        for pt in self.info_dict_16s():
            c.append(np.sum(np.vstack([self.info_dict_16s[pt][t]['16s']
                                        for t in self.info_dict_16s[pt].keys()]), 0))
        cts = np.vstack(c).T
        cts[cts > 0] = 1
        counts = np.sum(cts, 1)

        filt_out = np.where(counts > val)[0]

        self.make_16s_dict(np.array(self.data_sm), idxs=filt_out)
        return filt_out


    def amt_over_time(self, mols, dict_name=None, pts=None, save = None, filename = None):
        mapping = {'Cleared': 'green', 'Recur': 'red'}
        if pts is None:
            pts = self.pt_info_dict.keys()
        for molecule in mols:
            # plt.figure()
            fig, ax = plt.subplots()
            for patient in pts:
                if dict_name is None:
                    tmpts = list(self.pt_info_dict[patient].keys())
                    data = [self.pt_info_dict[patient][t]['DATA'][molecule] for t in tmpts]
                    label = [self.pt_info_dict[patient][t]
                            ['PATIENT STATUS (BWH)'] for t in tmpts]
                else:
                    tmpts = list(dict_name[patient].keys())
                    data = [dict_name[patient][t]['DATA'][molecule]
                            for t in tmpts]
                    label = [dict_name[patient][t]
                            ['PATIENT STATUS (BWH)'] for t in tmpts]

                from matplotlib.lines import Line2D
                custom_lines = [Line2D([0], [0], color='g', lw=4),
                                Line2D([0], [0], color='r', lw=4)]

                
                ax.legend(custom_lines, ['Cleared', 'Recur'],fontsize=12)

                labels = [colors.to_rgb(mapping[l]) for l in label]
                ax.scatter(tmpts, data, color=labels, linewidth=5)
                ax.plot(tmpts, data, '--k', linewidth=.5)
                ax.set_yscale('log')
                plt.xlabel('Week',fontsize=15)
                plt.ylabel('Amt',fontsize = 15)
                plt.title(molecule,fontsize=18)
                plt.xticks(fontsize=12)
                plt.yticks(fontsize=12)
            filename = 'lineplot_' + molecule + '.pdf'
            if save:
                plt.savefig(filename)
            plt.show()

