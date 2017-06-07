import sys
import math
import heapq
import numpy as np
import itertools
from scipy.sparse import csc_matrix
from collections import defaultdict
#global document_count
np.set_printoptions(threshold=np.nan)

def check_word_across_docs(word,word_across_docs):
	if (int(word)-1) not in word_across_docs:
		word_across_docs[int(word)-1]=1
	else:
		word_freq_across_docs=word_across_docs.get(int(word)-1)
		word_freq_across_docs+=1
		word_across_docs[int(word)-1]=word_freq_across_docs
	return word_across_docs

def calculate_idf(document,word,word_across_docs,document_count):
	n=int(document_count)
	df=float(word_across_docs.get(int(word)))
	takelogof=float(n+1) / (df+1)
	idf=math.log(takelogof,2)
	return idf

def read_input(input_file,documents_metastore,word_across_docs,total_docs):
	nextline=input_file.readline()
	while (nextline):
		nextline=nextline.rstrip()
		break_input_line=nextline.split(" ")
		key=int(break_input_line[0])
		word_across_docs=check_word_across_docs(break_input_line[1],word_across_docs)
		value=(int(break_input_line[1])-1,break_input_line[2])
		if (key-1) not in documents_metastore:
			value_list=[]
			value_list.append(value)
			documents_metastore[key-1]=value_list
		else:
			value_list=documents_metastore.get(key-1)
			value_list.append(value)
			documents_metastore[key-1]=value_list
		nextline=input_file.readline()
	maxWords=max(word_across_docs.keys())+1
	vector_list=csc_matrix((total_docs,maxWords), dtype=float)


	return documents_metastore,word_across_docs,vector_list

def get_unit_vector(document,documents_metastore,word_across_docs,document_count,vocab_words,total_words,vector_list,merged_vectors_map,magnitude):
	doc_info=documents_metastore.get(document)
	c_squared_tf_idf=0
	insidecount=0
	vector_list=vector_list.toarray()
	#vector_tf_idf={}
	for word in doc_info:
		insidecount+=1
		tf=int(word[1])
		idf=calculate_idf(document,word[0],word_across_docs,document_count)
		tf_idf=tf*idf
		square_tf_idf=tf_idf*tf_idf
		c_squared_tf_idf=c_squared_tf_idf+square_tf_idf
		#print tf_idf
		vector_list[int(document),int(word[0])]=tf_idf
		merged_vectors_map[int(document)]=int(document)
	euclidean_distance=c_squared_tf_idf**(1/2.0)
	magnitude[int(document)]=euclidean_distance
	size=vector_list.shape
	#size=max(vector_tf_idf.keys())+1
	for col in range(0,size[1]):
		tfidf=vector_list[int(document),col]
		normalised_tfidf=float(tfidf)/euclidean_distance
		vector_list[int(document),col]=normalised_tfidf
	return csc_matrix(vector_list),merged_vectors_map,magnitude

def calculate_cosine_similarity(a,b,mag_a,mag_b,similar_words):
	sop=0
	for i in similar_words:
		dot_product=a[0,i]*b[0,i]
		sop=sop+dot_product
	tot_mag=float(mag_a*mag_b)
	return "{:.5f}".format(float(sop)/tot_mag)

def check_similar_words(a,b,documents_metastore):
	doc_info1=documents_metastore.get(a)
	doc_info2=documents_metastore.get(b)
	doc_info1=zip(*doc_info1)

	doc_info1=list(doc_info1[0])
	doc_info1 = list(map(int, doc_info1))

	doc_info2=zip(*doc_info2)
	doc_info2=list(doc_info2[0])
	doc_info2 = list(map(int, doc_info2))

	similar_word_list=list(set(doc_info1)&set(doc_info2))
	return similar_word_list

def convert_to_one_cluster(merged_vectors):
	if isinstance(merged_vectors[0], int):
		tup1=(merged_vectors[0],)
	else:
		tup1=tuple(merged_vectors[0])


	if isinstance(merged_vectors[1], int):
		tup2=(merged_vectors[1],)
	else:
		tup2=tuple(merged_vectors[1])

	return tup1+tup2





def calculate_centroid(merged_vectors,vector_list):
	all_vectors=[]
	v_clustered=[]
	for i in range(len(merged_vectors)):
		v_temp=vector_list[int(merged_vectors[i])].toarray()
		
		
		#print v_temp
		all_vectors.append(v_temp[0])

	for i in range(len(all_vectors[0])):
		sum=0
		for j in range(len(merged_vectors)):
			temp_vect=all_vectors[j]
			sum=sum+temp_vect[i]
		v_clustered.append(float(sum)/len(merged_vectors))
	return v_clustered



def take_average(merged_vectors,documents_metastore,vector_list,merged_vectors_map,new_merged_vectors_map,document_count):
	#print "working for ",merged_vectors
	merged_vectors=convert_to_one_cluster(merged_vectors)
	#print "clustered as ",merged_vectors
	v12=calculate_centroid(merged_vectors,vector_list)

	value_list=[]
	for i in range(len(v12)):
		if v12[i] !=0.0:
			value=(i,1)
			value_list.append(value)
	documents_metastore[merged_vectors]=value_list
	size=vector_list.shape
	vector_list=vector_list.toarray()
	vector_list = np.vstack([vector_list,v12])
	document_count+=1
	vector_list=csc_matrix(vector_list)
	new_merged_vectors_map[merged_vectors]=size[0]
	merged_vectors_map[size[0]]=merged_vectors
	#print "Returnng"
	return documents_metastore,vector_list,merged_vectors_map,new_merged_vectors_map,v12,document_count


def calculate_other_cs(merge_docs,vector_list,merged_vectors_map,new_merged_vectors_map,documents_metastore,input_heap,clusters):
	size=vector_list.shape
	for i in range(size[0]):
		if i in merged_vectors_map:
			already_merged=merged_vectors_map.get(i)
			if isinstance(already_merged, int):
				already_merged=(already_merged,)
			else:
				already_merged=tuple(already_merged)
			if tuple(already_merged) not in clusters or tuple(already_merged) == merge_docs:
				pass
			else:
				vector_pair=(merge_docs,merged_vectors_map.get(i))
				a=vector_list[new_merged_vectors_map.get(merge_docs)]
				a=a.toarray()
				x = np.array(a)
				mag_a=np.linalg.norm(x)
				if isinstance(merged_vectors_map.get(i), int):
					b=vector_list[merged_vectors_map.get(i)]
					b=b.toarray()
					x = np.array(b)
					mag_b=np.linalg.norm(x)
				else:
					b=vector_list[i]
					b=b.toarray()
					x = np.array(b)
					mag_b=np.linalg.norm(x)
				similar_words=check_similar_words(merge_docs,merged_vectors_map.get(i),documents_metastore)
				if len(similar_words)!=0:
					cos_similarity=calculate_cosine_similarity(a,b,mag_a,mag_b,similar_words)
					if cos_similarity !=0.0:
						#print "pushing -> cosine val :  ",cos_similarity," Pair: ",vector_pair
						val=(1.0-float(cos_similarity),vector_pair)
						#print "pushing new pair ",val
						heapq.heappush(input_heap,val)
						heapq.heapify(input_heap)
	return vector_list,input_heap


def already_clustered(current_docs,popped_clusters):

	if isinstance(current_docs[0], int):
		tup1=(current_docs[0],)
	else:
		#print merge_docs[0]
		tup1=tuple(current_docs[0])


	#if len(merge_docs[1])>1:
	if isinstance(current_docs[1], int):
		tup2=(current_docs[1],)
	else:
		tup2=tuple(current_docs[1])

	if tup1 not in popped_clusters and tup2 not in popped_clusters:
		return True
	else:
		return False


def add_one(tup):
	#print "Value ",len(tup)
	final_val=""
	if len(tup)==1:
		val=int(tup[0])+1
		final_val=val
	else:
		for val in tup:
			val=int(val)+1
			final_val=final_val+ str(val)+","
		final_val=final_val[:-1]
	return final_val



s=sys.argv
input_file=open(s[1],"r")
k=s[2]
#global document_count
document_count=input_file.readline()
vocab_words=input_file.readline()
total_words=input_file.readline()
documents_metastore={}
word_across_docs={}
magnitude={}
merged_vectors_map={}
documents_metastore,word_across_docs,vector_list=read_input(input_file,documents_metastore,word_across_docs,int(document_count))
outsidecount=0
for document in documents_metastore.keys():
	outsidecount+=1
	vector_list,merged_vectors_map,magnitude=get_unit_vector(document,documents_metastore,word_across_docs,document_count,vocab_words,total_words,vector_list,merged_vectors_map,magnitude)



## Run cosine similarity
cosine_similarities=[]
for vector_pair in itertools.combinations(range(0,int(document_count)),2):
	a=vector_list[int(vector_pair[0])]
	a=a.toarray()
	b=vector_list[int(vector_pair[1])]
	b=b.toarray()
	similar_words=check_similar_words(int(vector_pair[0]),int(vector_pair[1]),documents_metastore)
	#print similar_words
	if len(similar_words)!=0:
		x = np.array(a)
		mag_a=np.linalg.norm(x)
		#print a,mag_a

		x = np.array(b)
		mag_b=np.linalg.norm(x)
		#print b,mag_b

		cos_similarity=calculate_cosine_similarity(a,b,mag_a,mag_b,similar_words)
		if cos_similarity !=0.0:
			val=(1.0-float(cos_similarity),vector_pair)
			cosine_similarities.append(val)
input_heap=[]
for tup in cosine_similarities:
	heapq.heappush(input_heap,tup)





################### Heap Process and Clustering ################## 
dum=[]
count=0
v_0=vector_list[0].toarray()
for i in v_0[0]:
	#print i
	if i != 0.0:
		zz=(count,i)
		dum.append(zz)
	count+=1
dum1=[]
count=0
v_1=vector_list[1].toarray()
for i in v_1[0]:
	if i != 0.0:
		zz=(count,i)
		dum1.append(zz)
	count+=1
new_merged_vectors_map={}
clusters = [ (a,) for a in range(0,int(document_count))]
#print clusters 
popped_clusters=[]
while(True):
	merge_docs=heapq.heappop(input_heap)
	dummy=merge_docs
	merge_docs=merge_docs[1]
	if(already_clustered(merge_docs,popped_clusters)):
		documents_metastore,vector_list,merged_vectors_map,new_merged_vectors_map,merged_vector,document_count=take_average(merge_docs,documents_metastore,vector_list,merged_vectors_map,new_merged_vectors_map,int(document_count))


		if isinstance(merge_docs[0], int):
			tup1=(merge_docs[0],)
		else:
			#print merge_docs[0]
			tup1=tuple(merge_docs[0])

		if isinstance(merge_docs[1], int):
			tup2=(merge_docs[1],)
		else:
			tup2=tuple(merge_docs[1])
		clusters.remove(tup1)
		clusters.remove(tup2)
		popped_clusters.append(tup1)
		popped_clusters.append(tup2)
		c=tuple(list(tup1+tup2))
		clusters.append(c) ## append like (43,85,12)

		if len(clusters) == int(k):
			break

		vector_list,input_heap=calculate_other_cs(c,vector_list,merged_vectors_map,new_merged_vectors_map,documents_metastore,input_heap,clusters)
		#print clusters

#print "Final : ",clusters
for tup in clusters:
	tup=add_one(tup)
	print tup










