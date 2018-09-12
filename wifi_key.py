#-*- coding: utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding('utf8')
import operator

#搜索两个字符串里的最长连续子串，返回最长子串及其长度
def find_lccs_substr(s1, s2): 
	m=[[0 for i in range(len(s2)+1)]  for j in range(len(s1)+1)]  #生成0矩阵，为方便后续计算，比字符串长度多了一列
	mmax=0   #最长匹配的长度
	p=0  #最长匹配对应在s1中的最后一位
	for i in range(len(s1)):
		for j in range(len(s2)):
			if s1[i]==s2[j]:
				m[i+1][j+1]=m[i][j]+1
				if m[i+1][j+1]>mmax:
					mmax=m[i+1][j+1]
					p=i+1
	if s1:
		return s1[p-mmax:p].decode('utf-8')#,mmax   #返回最长子串及其长度

#取出topN value的key的list 
def find_topN_value_key(dictA,topn):
    list_N =[]
    if dictA:
        for i in range(0,topn):
            max_value_key = max(dictA.iteritems(), key=operator.itemgetter(1))[0]
            list_N.append(max_value_key)
            dictA.pop(max_value_key)  #删除最大的key,value，方便下一次循环
            if not dictA:
                break
    return list_N

#抽取一个dict的ssid中的key_string
def extract_keywords(dict_ssid,topn):  #list_ssid is a dict like {ssid1:cnt1,ssid2:cnt2,……}
    # length = len(list_ssid)
    # dict_key_cnt = {}  #{key1:cnt1,key2:cnt2,……}
    # for i in range(0,length-1): #1->2,2->3,n-1->n
    #     for j in range(i+1,length):
    #         lscc_result = find_lccs_substr(list_ssid[i],list_ssid[j])
    #         key_ssid = lscc_result[0]  #最长连续子串
    #         #key_length = lscc_result[1]  #最长连续资产的长度
    #         if len(key_ssid)>2:
    #             if key_ssid in dict_key_cnt.keys():
    #                 dict_key_cnt[key_ssid] = dict_key_cnt[key_ssid] + 1
    #             else : 
    #                 dict_key_cnt[key_ssid] = 1

    list_ssid = dict_ssid.keys()  #取出某shop的所有ssid list
    length = len(list_ssid)
    dict_key_cnt = {}  #{key1:cnt1,key2:cnt2,……}
    for i in range(0,length-1): #1->2,2->3,n-1->n
        for j in range(i+1,length):
            lscc_result = find_lccs_substr(list_ssid[i],list_ssid[j])
            if lscc_result:
                key_ssid = lscc_result[0]  #最长连续子串
                #key_length = lscc_result[1]  #最长连续资产的长度
                if len(key_ssid)>2:
                    if key_ssid in dict_key_cnt.keys():
                        m = dict_ssid[list_ssid[i]]
                        n = dict_ssid[list_ssid[j]]
                        dict_key_cnt[key_ssid] = int(dict_key_cnt[key_ssid]) + int(m) + int(n)
                    else : 
                        m = dict_ssid[list_ssid[i]]
                        n = dict_ssid[list_ssid[j]]
                        dict_key_cnt[key_ssid] = int(m)+int(n)
    #输出dict {string1:cnt1,string2:cnt2,……}
    # print 'output dict_key_cnt:'
    # for key, value in dict_key_cnt.iteritems():
    #     print key,value
    list_keys = find_topN_value_key(dict_key_cnt,topn)
    return list_keys

#抽取不同shop的key_string里频率高的string
def find_unused_fre_key(dict_key):
    list_fre = []
    for key,value in dict_key.items():
        for i in range(len(value)):
            list_fre.append(value[i])
    a = {}
    list_fre_key = []
    for i in list_fre:
        if list_fre.count(i)>1:
            a[i] = list_fre.count(i)
    for key,value in a.items():
        list_fre_key.append(key)
    print(list_fre_key)
    return list_fre_key

#过滤list成员，求差集，在B中但不在A中
def diff(listA,listB):
    retD = list(set(listB).difference(set(listA)))
    return retD

#常见路由器等无意义词
list_unused = ['TP-LINK','MERCURY','FAST','Tenda','Xiaomi','PHICOMM','D-Link','TPGuest','dlink','ChinaNet','CMCC','CU','B-LINK','USER','360WiFi',
        '客厅的父母乐','GAOKE','0x','dlink','netcore','netgear','androidap','d-link_dir-612','@ffan']

if __name__ == '__main__':
    DB_FN = 'F:\\wifi3.txt'
    with open(DB_FN, 'r') as f:
        data = f.readlines()
    dict_shop_ssid = {}
    for line in data:
        l = line.strip().split('|')
        ssid = l[0].strip().lower().replace(".","").replace(" ","").replace("_","").replace("-","")  #去掉空格，转成小写,去掉 空格 . _ -
        #print ssid
        shop = l[1]
        cnt = l[2]
        #print shop
        if shop in dict_shop_ssid.keys():
            temp_dict = dict_shop_ssid[shop]
            if ssid in temp_dict.keys():
                temp_dict[ssid] = int(temp_dict[ssid]) + int(cnt)
            else :
                temp_dict[ssid] = cnt
        else:
            temp = {}
            temp[ssid] = cnt
            dict_shop_ssid[shop] = temp
    #输出dict {shop1:{ssid1:cnt1,ssid2:cnt2,……}, shop2:...}
    # for key, value in dict_shop_ssid.iteritems():
    #     print key,value

    #第一次粗略计算的keys
    print 'first output result:'
    dict_key_all = {}
    for shop in dict_shop_ssid.keys():
        temp_dict = dict_shop_ssid[shop]
        temp_dict_keys = extract_keywords(temp_dict,2)
        dict_key_all[shop] = temp_dict_keys
        print shop,dict_key_all[shop]
    #输出dict {shop1:[ssid1,ssid2,……], shop2:...}
    # print 'output dict_key_all:'
    # for key, value in dict_key_all.iteritems():
    #     print key,value

    #第二次计算，过滤掉shop间的高频关键字
    list_fre_keys = find_unused_fre_key(dict_key_all)  #shop间的高频关键词
    for key,value in dict_key_all.items():
        dict_key_all[key] = diff(list_fre_keys,dict_key_all[key])  #在dict_key_all[key]过滤掉高频关键词

    #第三次计算，过滤掉路由器
    #list_unused 常用路由器list,转小写
    for i in range(len(list_unused)):
        list_unused[i] = list_unused[i].lower().replace(".","").replace(" ","").replace("_","").replace("-","")
    for key,value in dict_key_all.items():
        dict_key_all[key] = diff(list_unused,dict_key_all[key])  #在dict_key_all[key]过滤掉常用路由器list

    #dict_key_all为最后结果
    #输出dict {shop1:[ssid1,ssid2,……], shop2:...}
    print 'output dict_key_all:'
    for key, value in dict_key_all.iteritems():
        print key,value
    print 'save result to a txt:'
    fileObject = open('keywords_1.txt','w')
    for key in dict_key_all:
        fileObject.write(key)
        fileObject.write('|')
        fileObject.write(str(dict_key_all[key]))
        fileObject.write('\n')
    fileObject.close()
    



        