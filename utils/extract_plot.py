import os
import sys
import matplotlib.pyplot as plt

#filename = '1_layer_16_unit_log.txt'
#filename = '2_layer_128_unit_gamma_point95.txt'
#filename = 'log2.txt'

def parse_file(filename):
    infile = file(filename)
    parsed_filename = "parsed_" + filename.split("/")[-1]

    parsedopen = open(parsed_filename, 'w')

    write_next = True
    lines_seen = set()  # holds lines already seen
    for line in infile:
        if 'Batch' in line and line not in lines_seen:
            lines_seen.add(line)
            line = line.replace(":", "")
            line = "\t".join(line.split()[3:4]) + " "
            parsedopen.write(line)
            write_next = True
        if 'Average' in line and line not in lines_seen and write_next == True:
            line = line.replace(":", "")
            parsedopen.write("\t".join(line.split()[4:5]) + "\n")
            lines_seen.add(line)
            write_next = False

    infile.close()
    parsedopen.close()
    return parsed_filename

if __name__ == '__main__':
    colors = ['blue', 'green', 'sandybrown']
    for i, filename in enumerate(sys.argv):
        if i == 0: continue
        filename = sys.argv[i]
        parsed = parse_file(filename)
        x = []
        y = []
        parsed_file = open(parsed)
        for line in parsed_file:
            fields = line.strip().split()
            x.append(fields[0])
            y.append(fields[1])
        parsed_file.close()
        plt.plot(x, y, color=colors[i-1], marker="o", linestyle="solid", label=filename, linewidth=2, markersize=3)

    # minx = 0
    # maxx = 200
    # miny = -70000  #-160000    #max(min(y),-70000)
    # maxy = 0  #2000   #max(max(y),0)

    # plt.gca().invert_yaxis()
    plt.xlabel('Batch number')
    plt.ylabel('Average reward')
    plt.title('Multiagent simple_spread training with PG with 1 layer and 16 hidden units')
    # plt.axis([minx,maxx,miny,maxy])
    plt.show()
