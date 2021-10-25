import argparse
import os
import glob

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('run_name', type=str)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    dirname = os.path.join("runs", args.run_name, 'eval')
    results={}
    for file in glob.glob(os.path.join(dirname, "*.txt")):
        split = file.split('-')
        dataset = split[1]
        append_missing = (len(split) > 2 and split[2][:2] == 'am')
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
    
    outfile = os.path.join("runs", args.run_name, "eval-results.txt")
    with open(outfile, 'w') as writer:
        for dataset in results.keys():
            writer.write(dataset+":\n")
            for baseline in results[dataset].keys():
                writer.write("\t%s:\n" % baseline)
                for key, val in results[dataset][baseline].items():
                    writer.write("\t\t%s: %d\n" % (key, val))
        

