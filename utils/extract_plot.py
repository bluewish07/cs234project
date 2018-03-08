import os
import sys
import matplotlib.pyplot as plt

#filename = '1_layer_16_unit_log.txt'
#filename = '2_layer_128_unit_gamma_point95.txt'
#filename = 'log2.txt'
filename = sys.argv[1]

infile = file(filename)

newopen = open('new.txt', 'w')

write_next = True
lines_seen = set() # holds lines already seen
for line in infile :
    if 'Batch' in line and line not in lines_seen:
        lines_seen.add(line)
        line = line.replace(":", "") 
        line = "\t".join(line.split()[3:4]) + " "
        newopen.write(line)
        write_next = True
    if 'Average' in line and line not in lines_seen and write_next == True:
        line = line.replace(":", "") 
        newopen.write( "\t".join(line.split()[4:5]) + "\n")
        lines_seen.add(line)
        write_next = False

infile.close()
newopen.close()

x = []
y = []
file = open('new.txt')
for line in file:
    fields = line.strip().split()
    x.append(fields[0])
    y.append(fields[1])
file.close()

minx = 0
maxx = 200 
miny = -70000  #-160000    #max(min(y),-70000)
maxy = 0  #2000   #max(max(y),0)
plt.plot(x,y, 'go-', label='line 1', linewidth=2)
plt.gca().invert_yaxis()
plt.xlabel('Batch number')
plt.ylabel('Average reward')
plt.title('Multiagent simple_spread training with PG with 1 layer and 16 hidden units')
plt.axis([minx,maxx,miny,maxy])
plt.show()
