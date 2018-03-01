from __future__ import print_function
import numpy as np
from matplotlib import pyplot as plt
import sys
import os
import itertools
from matplotlib import gridspec
import glob
import codecs
from iso639 import languages
plt.switch_backend('agg')

lang_id_map = {
    "000": "English",
    "102": "Assamese",
    "103": "Bengali",
    "104": "Pashto",
    "105": "Turkish",
    "106": "Tagalog",
    "107": "Vietnamese",
    "201": "Haitian Creole",
    "202": "Swahili (individual language)",
    "203": "Lao",
    "204": "Tamil",
    "205": "Northern Kurdish",
    "206": "Zulu",
    "207": "Tok Pisin",
    "301": "Cebuano",
    "302": "Kazakh",
    "303": "Telugu",
    "304": "Lithuanian",
    "305": "Guarani",
    "306": "Igbo",
    "307": "Amharic",
    "401": "Mongolian",
    "402": "Javanese",
    "403": "Dholuo",
    "404": "Georgian"
}

# CONSTANTS
diff_type="symb_er"
budget_type = "budget"
ref_type="Random"
idxs = [40, 500, 2000, 5000]

def main():
    langs = {}
    lang_num_words = {}
   
    # READS IN ALL DATA    
    # sys.argv[2] = num_words.txt 
    with open("./num_words.txt", "r") as f:
        for l in f:
            key, val = l.strip().split(None, 1)
            lang_num_words[key] = int(val)
    
    for d in glob.glob("results/*/*.txt"):
        lang_id, lang = os.path.basename(os.path.dirname(d)).split("_", 1)
        exp_type = os.path.splitext(os.path.basename(d))[0]
        
        var = readdata(d, lang)
        var["budget_norm"] = var["budget"] / lang_num_words[lang_id]
        
        try:
            langs[lang_id][exp_type] = var
        except KeyError:
            langs[lang_id] = {exp_type: var}
    
    for l in langs.iterkeys():
        langs[l]["num_words"] = lang_num_words[l]
        for var1 in langs[l].keys():
            for var2 in langs[l].keys():
                diff_name = "diff-" + var1 + "-" + var2
                try:
                    diff = (langs[l][var2][diff_type] - langs[l][var1][diff_type]) / langs[l][var2][diff_type]
                    langs[l][var1][diff_name] = diff
                except:
                    pass

    
    ###########################################################################
    # Compute Scores 
    ###########################################################################
    average1 = np.zeros(langs["000"]["FC-4+"]["budget"].shape)
    average2 = np.zeros(langs["000"]["FC-4+"]["budget"].shape)
    
    num_langs = 0
    diff1_name = "diff-FC-4+-Random"
    diff2_name = "diff-FC-4-Random"
    exp1_name = "FC-4+"
    exp2_name = "FC-4"
    for l, results in langs.iteritems():
        try:
            average1 += results[exp1_name][diff1_name]
            average2 += results[exp2_name][diff2_name]
            num_langs += 1
        except KeyError:
            print("Key Error: ", l)
        
    
    average1 /= num_langs
    average2 /= num_langs

    ###########################################################################
    # Compute scores 2
    ###########################################################################
    scores1 = {}
    scores2 = {}
    scoresR = {}
    
    
    for l, results in langs.iteritems():
        try:
            scores1[l] = {}
            scores2[l] = {}
            scoresR[l] = {}
            for i_e in range(len(idxs)):
                for i, v in enumerate(results["Random"]["budget_letters"]):
                    if v >= idxs[i_e] and v < idxs[i_e + 1]: 
                        try:
                            scores1[l][idxs[i_e]].append(results[exp1_name][diff_type][i])
                            scores2[l][idxs[i_e]].append(results[exp2_name][diff_type][i])
                            scoresR[l][idxs[i_e]].append(results["Random"][diff_type][i])
                        except KeyError:
                            scores1[l][idxs[i_e]] = [results[exp1_name][diff_type][i]]
                            scores2[l][idxs[i_e]] = [results[exp2_name][diff_type][i]]
                            scoresR[l][idxs[i_e]] = [results["Random"][diff_type][i]]
                
                scoresR[l][idxs[i_e]] = np.mean(scoresR[l][idxs[i_e]])
                scores1[l][idxs[i_e]] = np.mean(scores1[l][idxs[i_e]])
                scores2[l][idxs[i_e]] = np.mean(scores2[l][idxs[i_e]])
                
        except KeyError:
            print("Key Error: ", l)



    f, ax = plt.subplots(len(idxs), 1, figsize=(15, 5), sharex=True)
    for i_ax, seed_size in enumerate(idxs):
        # plt.bar(range(len(lang_scores1.items())), sorted(lang_scores1.itervalues()), width=0.4, label="FC-4+")
        # plt.bar(0.4 + np.array(range(len(lang_scores1.items()))), [lang_scores2[k] for k,v in sorted(lang_scores1.iteritems(), key=lambda (k, v): (v, k))], width=0.4, label="FC-4", alpha=0.4, edgecolor='k')
        # #plt.bar(0.4 + np.array(range(len(lang_scores1.items()))), [lang_scoresR[k] for k,v in sorted(lang_scores1.iteritems(), key=lambda (k, v): (v, k))], width=0.2, label="Random", alpha=0.4, edgecolor='k')
        # plt.xticks(range(25), [languages.name[lang_id_map[k]].part3 for k,v in sorted(lang_scores1.iteritems(), key=lambda (k, v): (v, k))], fontsize=14)
        # plt.grid(alpha=0.6)
        # plt.legend(loc="upper left", fontsize=16)
        # plt.xlabel("ISO 639-3 Language Code", fontsize=16)
        # plt.ylabel("Relative Improvement (SER)", fontsize=16)
        # #plt.ylim([0.0, 1.0])
        # plt.axhline(linewidth=1, color='k')
        # #plt.savefig("../Results/figures/language_relative_improvement_WER.png")
        order1 = [(k, v[seed_size]) for k, v in sorted(scores1.iteritems(), key=lambda (k, v): (v[seed_size], k))]
        order2 = [(k, scores2[k][seed_size]) for k, v in sorted(scores1.iteritems(), key=lambda (k, v): (v[seed_size], k))]
        orderR = [(k, scoresR[k][seed_size]) for k, v in sorted(scores1.iteritems(), key=lambda (k, v): (v[seed_size], k))]
    
        ax[i_ax].bar(range(len(order1)), [v for k, v in order1], width=0.2, label="FC-4+")
        ax[i_ax].bar(0.2 + np.array(range(len(scores1.items()))), [v for k, v in order2], width=0.2, label="FC-4", alpha=0.4, edgecolor='k')
        ax[i_ax].bar(0.4 + np.array(range(len(scores1.items()))), [v for k, v in orderR], width=0.2, label="Random")
    
        ax[i_ax].set_xticks(range(25))
        ax[i_ax].set_xticklabels([languages.name[lang_id_map[k]].part3 for k, v in order1], fontsize=14)
    
        ax[i_ax].grid(alpha=0.6)
        ax[i_ax].legend(loc="upper left", fontsize=16)
        ax[i_ax].set_ylim([0.0, 100.0])
        #plt.axhline(linewidth=1, color='k')
        ax[i_ax].set_xlabel("ISO 639-3 Language Code", fontsize=16)
        ax[i_ax].set_ylabel("(PER)", fontsize=16)

        plt.savefig("./images/lang_improvement_per.png")


    xticks = langs["000"]["Random"]["budget"]
    plt.plot(xticks, average1, label="FC-4+")
    plt.plot(xticks, average2, label="FC-4", ls="--")
    plt.xscale("log")
    plt.xticks([j for i, j in enumerate(xticks) if i % 4 == 0],
                       ["{:0.0f}".format(j) for i, j in enumerate(xticks) if i % 4 == 0])
    plt.legend(loc="lower right", fontsize=16)
    plt.xlabel("# Words in Seed Lexicon", fontsize=14)
    plt.ylabel("Relative Gain (Phone Accuracy)", fontsize=14)
    #plt.title("Mean Relative Improvement Across All Languages", fontsize=18)
    plt.grid() 
    plt.savefig("./images/avg_improvement.png")



def readdata(filename, lang):
    with open(filename, "r") as f:
        data = {}
        headers = f.readline().split()
        for l in f:
            for field_idx, field_val in enumerate(l.split()):
                try:
                    data[headers[field_idx]].append(float(field_val))
                except KeyError:
                    data[headers[field_idx]] = [float(field_val)]
        
        for k, v in data.iteritems():
            data[k] = np.asarray(v)
        
        data["name"] = os.path.splitext(os.path.basename(filename))[0]
        if data["name"] in ("Random"):
            data["ls"] = "--"
        else:
            data["ls"] = "-"
        
        data["name"] = data["name"] + " " + lang
    
    return data


if __name__ == "__main__":
    main() 

