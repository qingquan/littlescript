
import re
from math import log,exp,sqrt
import numpy as np
from numpy import linalg as LA

def getLineDetails(line):
	items = line.split(' ',1)
	loc = [int(i) for i in items[0].split(',')]
	rssVec = dict((i[0],float(i[1]))for i in re.findall(r'(\w+):(-\d+\.\d+)',items[1]))
	return loc,rssVec
def getRefData(ref_file):
	lines = [l for l in open(ref_file) if ',' in l]
	result = [getLineDetails(i) for i in lines]
	print '%s len: %d items found'%(ref_file,len(result))
	return result
def calcSim(r1,r2,a=1.9,b=10.0):
	s = set(r1.keys())|set(r2.keys())
	size = len(s)
	v1 = np.zeros(size,dtype=np.float64)
	v2 = np.zeros(size,dtype=np.float64)
	for i,did in enumerate(s):
		if did in r1: v1[i] = a**(float(r1[did])*1.0/b)
		if did in r2: v2[i] = a**(float(r2[did])*1.0/b)
	norm1,norm2 = LA.norm(v1),LA.norm(v2)
	sim = np.dot(v1,v2)/norm1/norm2
	if np.isnan(sim): return 0.0
	else: return sim
def getRSSILines(rssi_file, out_file):
    simList = []
    rssVecList = []
    rssi_file = open(rssi_file, 'r')
    rssi_list = rssi_file.readlines()
    for k in range(len(rssi_list)):
        rssVec = dict((i[0],float(i[1]))for i in re.findall(r'(\w+):(-\d+\.\d+)',rssi_list[k]))
        rssVecList.append(rssVec)
        if k>=2:
            sim = calcSim(rssVecList[k-1], rssVecList[k])
            simList.append(sim)
            print sim
    to_file = open(out_file, 'w')
    for word in simList:
        to_file.write(str(word)+'\n')
    to_file.close()
    #print rssVecList

def final_selection(sims,points_ref,n=15):
	idx = sorted(range(len(sims)),key=lambda x:sims[x])
	top10 = [sims[i] for i in idx[-1*n:]]

	thres = np.mean(top10)+0.55*np.std(top10)
	# diffs = [abs(sims[i+1]-sims[i]) for i in xrange(n-1)]
	# thres = top10[diffs.index(max(diffs))]

	candidates = [exp(top10[i]) for i in xrange(n) if top10[i]>thres]
	N = sum(candidates)
	for i in xrange(len(candidates)): candidates[i] /= N
	points = [points_ref[i] for i in idx[-1*len(candidates):]]
	x,y = 0,0
	# print len(candidates)
	for i in xrange(len(candidates)):
		x += candidates[i]*float(points[i][0])
		y += candidates[i]*float(points[i][1])
	return x,y

def self_test(f,a=1.9,b=10):
	ref_data = getRefData(f)
	ref_pts = [i[0] for i in ref_data]
	y_cnt,n_cnt=0,0
	err = []
	for i in xrange(len(ref_data)):
		loc = ref_data[i][0]
		sims = []
		for j in xrange(len(ref_data)):
			if i==j: continue
			sims.append(calcSim(ref_data[i][1],ref_data[j][1],a,b))
		x,y = final_selection(sims,ref_pts)
		e = sqrt((x-float(loc[0]))**2+(y-float(loc[1]))**2)
		# print loc,'-'*5,e,'-'*5,x,y
		err.append(e)
		if e<167: y_cnt += 1
		else: n_cnt += 1
	print np.mean(err),np.std(err),y_cnt*1.0/(y_cnt+n_cnt)

def cos_test(ref_file,tar_file):
	ref_data = getRefData(ref_file)
	ref_pts = [i[0] for i in ref_data]
	tar_data = getRefData(tar_file)
	# tar_pts = [i[0] for i in tar_data]
	y_cnt,n_cnt=0,0
	err = []
	for i in xrange(len(tar_data)):
		loc = tar_data[i][0]
		sims = []
		for j in xrange(len(ref_data)):
			sims.append(calcSim(tar_data[i][1],ref_data[j][1]))
		x,y = final_selection(sims,ref_pts)
		e = sqrt((x-float(loc[0]))**2+(y-float(loc[1]))**2)
		# print loc,'-'*5,e,'-'*5,x,y
		err.append(e)
		if e<167: y_cnt += 1
		else: n_cnt += 1
		# with open('rrr.txt','a') as fout:
		# 	fout.write('%f\n'%e)
	print np.mean(err),np.std(err),y_cnt*1.0/(y_cnt+n_cnt)

print 'similarity for case1'
getRSSILines('rssi_data_case1.txt', 'rssi_data_case1_sim.txt')

print 'similarity for case2'
getRSSILines('rssi_data_case2.txt', 'rssi_data_case2_sim.txt')

print 'similarity for case3'
getRSSILines('rssi_data_case3.txt', 'rssi_data_case3_sim.txt')
#elf_test('z.avg.txt',2,10)
#elf_test('z.max.txt',2,10)
#elf_test('z.min.txt',2,10)
# cos_test('cut1_hex_p_ss.txt','cut1_hex_p_df.txt')

# fns = ['cut1_hex_p_df.txt','cut1_z_df.txt','cut1_hex_p_ss.txt','cut1_z_ss.txt','cut1_hex_p_one.s.txt','cut1_z_one.s.txt']
# for i in xrange(len(fns)):
# 	for j in xrange(len(fns)):
# 		if i==j: continue
# 		cos_test(fns[i],fns[j])
