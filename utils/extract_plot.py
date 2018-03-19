import math
import string
import sys
import matplotlib.pyplot as plt

#filename = '1_layer_16_unit_log.txt'
#filename = '2_layer_128_unit_gamma_point95.txt'
#filename = 'log2.txt'

def parse_file(filename, type="reward"):
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
        if ('Average '+type) in line and line not in lines_seen and write_next == True:
            number = line.split(":")[-1]
            number = number.split()[0]
            parsedopen.write(number + "\n")
            lines_seen.add(line)
            write_next = False

    infile.close()
    parsedopen.close()
    return parsed_filename

def parse_PG(filename, type="reward"):
    infile = file(filename)
    parsed_filename = "parsed_" + filename.split("/")[-1]

    parsedopen = open(parsed_filename, 'w')

    write_next = True
    lines_seen = set()  # holds lines already seen
    current_batch_number = None
    for line in infile:
        if 'Batch' in line and line not in lines_seen:
            lines_seen.add(line)
            line = line.replace(":", "")
            line = "\t".join(line.split()[3:4]) + " "
            current_batch_number = line
        elif 'Evaluating' in line and line not in lines_seen:
            lines_seen.add(line)
            parsedopen.write(current_batch_number)
            write_next = True
        elif ('Average '+type) in line and line not in lines_seen and write_next == True:
            line = line.replace(":", "")
            parsedopen.write("\t".join(line.split()[4:5]) + "\n")
            lines_seen.add(line)
            write_next = False

    infile.close()
    parsedopen.close()
    return parsed_filename

if __name__ == '__main__':
    # specify type here
    type = "reward"
    #type = "distance"

    colors = ['blue', 'green', 'sandybrown']
    for i, filename in enumerate(sys.argv):
        if i == 0: continue
        filename = sys.argv[i]
        parsed = None
        if "PG" in filename.split('/')[-1].split('.')[0].split('_'):
            parsed = parse_PG(filename, type=type)
        else:
            parsed = parse_file(filename, type=type)
        x = []
        y = []
        parsed_file = open(parsed)
        for line in parsed_file:
            fields = line.strip().split()
            x.append(fields[0])
            y.append(fields[1])
        parsed_file.close()
        label = string.join(parsed.split(".")[0].split("_")[1:], " ")
        plt.plot(x, y, color=colors[i-1], marker="o", linestyle="solid", label=label, linewidth=2, markersize=3)
        plt.legend(loc="lower right") # center right # lower right

    # minx = 0
    # maxx = 200
    # miny = -70000  #-160000    #max(min(y),-70000)
    # maxy = 0  #2000   #max(max(y),0)

    # plt.gca().invert_yaxis()
    plt.xlabel('Batch number')
    plt.ylabel('Average '+type)
    #plt.title('Multiagent simple_spread training with PG with 1 layer and 16 hidden units')
    # plt.axis([minx,maxx,miny,maxy])
    plt.show()

##### Below is for plotting action exploration range and std deviation

def parse_action_exploration(filename):
    infile = file(filename)
    parsed_filename = "parsed_" + filename.split("/")[-1]

    parsedopen = open(parsed_filename, 'w')

    min_dist = 100.
    max_dist = 0.
    std_dev_running_total = 0.
    skip = False
    clear = False
    timestep = 0
    batch_num = 0

    for line in infile:
        if 'action distance' in line:
            step = int(line.split(":")[-1])
            if step == 0 and clear:
                if batch_num % 5 == 0:
                    # write previous batch results
                    # if min_dist < 0.00001: min_dist = 0.
                    std_dev = math.sqrt(std_dev_running_total / 300.)
                    parsedopen.write(
                        str(timestep / 3) + "\t" + str(min_dist) + "\t" + str(max_dist) + "\t" + str(std_dev) + "\n")

                min_dist = 100.
                max_dist = 0.
                std_dev_running_total = 0.
                clear = False
                skip = False
            elif step == 99:
                clear = True
                batch_num += 1
            elif step >= 100:
                skip = True

            timestep += 1
        elif skip == False:
            dist = float(line)
            if dist < min_dist: min_dist = dist
            if dist > max_dist: max_dist = dist
            std_dev_running_total += math.pow(dist, 2)

    infile.close()
    parsedopen.close()
    return parsed_filename

# if __name__ == '__main__':
#
#     colors = ['blue', 'green', 'sandybrown']
#     labels = []
#     lines = []
#
#     for i, filename in enumerate(sys.argv):
#         if i == 0: continue
#         filename = sys.argv[i]
#         parsed = parse_action_exploration(filename)
#         x = []
#         min = []
#         max = []
#         std_dev = []
#         parsed_file = open(parsed)
#         for line in parsed_file:
#             fields = line.strip().split()
#             x.append(int(fields[0])+ (i-1)*50)
#             min.append(float(fields[1]))
#             max.append(float(fields[2]))
#             std_dev.append(float(fields[3]))
#         parsed_file.close()
#         label = string.join(parsed.split(".")[0].split("_")[1:], " ")
#         # for range
#         plt.plot(x, min, color=colors[i - 1], marker="_", linestyle="solid", label=label, linewidth=0, markersize=7, markeredgewidth=1)
#         plt.plot(x, max, color=colors[i - 1], marker="_", linestyle="solid", label=label, linewidth=0, markersize=7, markeredgewidth=1)
#         line = plt.vlines(x, min, max, color=colors[i - 1], linewidth=3)
#         lines.append(line)
#         labels.append(label)
#
#         # for std dev
#         # plt.plot(x, std_dev, color=colors[i - 1], marker="o", linestyle="solid", label=label, linewidth=2, markersize=3)
#         # plt.legend(loc="upper right")  # center right # lower right
#
#     # for range
#     plt.legend(lines, labels, loc="upper right") # center right # lower right
#     miny = 0
#     maxy = 12
#     plt.ylabel('Range of effective Euclidean distance')
#     plt.gca().set_ylim([miny, maxy])
#
#     # for std dev
#     # plt.ylabel('Standard deviation of effective Euclidean distance')
#     # miny = 0
#     # maxy = 3.5
#     # plt.gca().set_ylim([miny, maxy])
#
#     plt.xlabel('Timestep')
#     plt.show()
