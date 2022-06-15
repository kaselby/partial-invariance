import argparse
import os
import glob
import csv

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('run_name', type=str)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    dirname = os.path.join("runs", args.run_name, 'eval')
    results={}
    for file in glob.glob(os.path.join(dirname, "*.txt")):
        append_missing = 'am' in file.split('/')[-1]
        if append_missing:
            dataset = file.split('-')[1]
        else:
            dataset = file.split('-')[1][:-4]

        with open(file, 'r') as reader:
            lines = reader.readlines()

        assert len(lines) % 2 == 0
        def get_val(line):
            return float(line.split(" ")[-1].strip())

        if not dataset in results:
            results[dataset] = {}
        for i in range(int(len(lines)/2)):
            j = 2*i
            baseline_name = lines[j].split(" ")[0]
            baseline_acc, baseline_prec = get_val(lines[j]), get_val(lines[j+1])
            if not baseline_name in results[dataset]:
                results[dataset][baseline_name] = {}
            if append_missing:
                results[dataset][baseline_name]['acc-am'] = baseline_acc
                results[dataset][baseline_name]['prec-am'] = baseline_prec
            else:
                results[dataset][baseline_name]['acc'] = baseline_acc
                results[dataset][baseline_name]['prec'] = baseline_prec

    csvfile = os.path.join("runs", args.run_name, "eval-results.csv")
    with open(csvfile, 'w') as file:
        csvwriter = csv.writer(file)
        csvwriter.writerow(["Dataset", "Baseline", "Acc", "Prec", "Acc-AM", "Prec-AM"])
        for dataset in results.keys():
            for baseline, baseline_results in results[dataset].items():
                headers = ['acc','prec','acc-am','prec-am']
                row = [dataset, baseline, *[baseline_results[key] if key in baseline_results else -1 for key in headers]]
                csvwriter.writerow(row)

    outfile = os.path.join("runs", args.run_name, "eval-results.txt")
    with open(outfile, 'w') as writer:
        for dataset in results.keys():
            writer.write(dataset+":\n")
            for baseline in results[dataset].keys():
                writer.write("\t%s:" % baseline)
                for key, val in results[dataset][baseline].items():
                    writer.write("\t%s: %f" % (key, val))
                writer.write('\n')
        

