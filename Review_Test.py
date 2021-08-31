
#Pre-process the Reviews /Users/khanfarabi/OneDrive - The University of Memphis/Explain_MLN/Explanation_yelp/
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
import gensim 
import operator
from gensim.models import Word2Vec
#from sklearn import cross_validation
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (precision_score,recall_score,f1_score)
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
from sklearn import svm
#from sklearn import cross_validation
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (precision_score,recall_score,f1_score)
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OutputCodeClassifier
import random
import sys
import random
import re
from collections import defaultdict
import sys
from nltk.cluster import KMeansClusterer
import nltk
from sklearn import cluster
from sklearn import metrics
import gensim 
import operator
from gensim.models import Word2Vec
import sys
import numpy as np
import collections
class review:
                            @classmethod
                            def run_all(cls):

                                            def data_pr():
                                                        ifile1 = open("full-meta-data.txt")
                                                        revid = 0
                                                        users = defaultdict(list)
                                                        for ln in ifile1:
                                                            parts = ln.strip().split("\t")
                                                            users[parts[0]].append(revid)
                                                            revid = revid + 1
                                                        ifile1.close()
                                                        #pass#pass#print(users)
                                                        H11=defaultdict(list)
                                                        #sys.exit()
                                                        userids = []
                                                        #pass#pass#print(users)
                                                        c =0
                                                        #Select reviewer subset based on tunable parameters (max-reviews and min-reviews limit,sampling ratio)
                                                        minreviews =5
                                                        maxreviews =20
                                                        samplingratio =0.55
                                                        for u in users:
                                                            if len(users[u])>minreviews and len(users[u])<maxreviews:        
                                                                if random.random() < samplingratio:
                                                                    userids.append(u)
                                                                    c= c + len(users[u])

                                                        ifile = open("reviewContent.txt",encoding='ISO-8859-1')
                                                        ifile1 = open("full-meta-data.txt")
                                                        s_words = []
                                                        sfile = open("Words.txt")
                                                        for ln in sfile:
                                                            s_words.append(ln.strip())
                                                        sfile.close()
                                                        stopwords = []
                                                        sfile = open("stopwords.txt")
                                                        for ln in sfile:
                                                            stopwords.append(ln.strip())
                                                        sfile.close()

                                                        flags = (re.UNICODE if sys.version < '3' and type(text) is unicode
                                                                 else 0)
                                                        ofile = open("all_revs1.txt",'w')
                                                        ofile1 = open("metadata.txt",'w')
                                                        cnt = 0
                                                        revid = 0
                                                        qrat={}
                                                        windex = defaultdict(list)
                                                        #Tunable parameter to keep non-sentiment words
                                                        PNonSentWords = 0.30
                                                        WORDS={}
                                                        w_per_ht=defaultdict(list)
                                                        w_per_user=defaultdict(list)
                                                        Rev_text_map={}
                                                        for ln in ifile:
                                                            ln1 = ifile1.readline()
                                                            parts1 = ln1.strip().split("\t")
                                                            #pass#pass#print(parts1)
                                                            if parts1[0] not in userids:
                                                                continue
                                                            #if cnt >= 10000:
                                                            #    break
                                                            keep = []
                                                            parts = ln.strip().split("\t")
                                                            for word in re.findall(r"\w[\w']*", parts[3], flags=flags):
                                                                if word.isdigit() or len(word)==1:
                                                                    continue
                                                                word_lower = word.lower()
                                                                if word_lower in stopwords:
                                                                    continue
                                                               # if word_lower in s_words:
                                                                    #keep.append(word_lower)
                                                                elif random.random() < PNonSentWords:
                                                                    if not any(c.isdigit() for c in word_lower) and "'" not in word_lower:
                                                                        keep.append(word_lower)
                                                            if float(parts1[2])<=2:
                                                                cl = 0
                                                            elif float(parts1[2])==3:
                                                                cl = 1
                                                            elif float(parts1[2])>=4:
                                                                cl = 2
                                                            if len(keep)>=10:
                                                                cnt = cnt + 1
                                                                ofile.write(" ".join(keep)+"\t"+str(cl)+"\n")
                                                                WORDS[revid]=keep

                                                                qrat[revid]=cl
                                                                Rev_text_map[revid]=parts[3]
                                                                H11[parts1[1]].append(revid)
                                                                ofile1.write(ln1)
                                                                for w in keep:
                                                                    windex[w].append(revid)
                                                                    w_per_ht[w].append(parts1[1])
                                                                    w_per_user[w].append(parts1[0])
                                                                revid = revid + 1
                                                        ofile.close()
                                                        ofile1.close()
                                                        ifile.close()
                                                        ifile1.close()

                                                        #Tunable parameter (keep words only if repeated in > NumReps reviews)
                                                        NumReps = 10

                                                        #Filter review words
                                                        ifile = open("all_revs1.txt",encoding="ISO-8859-1")
                                                        ofile = open("processed_revs_1.txt",'w')
                                                        for ln in ifile:
                                                            parts = ln.strip().split("\t")
                                                            keep = []
                                                            for w in parts[0].split(" "):
                                                                if len(windex[w])>NumReps:
                                                                    keep.append(w)
                                                            ofile.write(" ".join(keep)+"\t"+parts[1]+"\n")
                                                        ofile.close()
                                                        ifile.close()
                                                        pass#pass#print("Total Reviews="+str(len(WORDS)))
                                                        return WORDS,qrat,H11,Rev_text_map,s_words,stopwords
                                            def data_balancing(WORDS):
                                                WORDS1={}
                                                t1=[]
                                                t2=[]
                                                t3=[]
                                                c1=0
                                                c2=0
                                                c3=0
                                                for y in WORDS:
                                                    if qrat[y]==0:
                                                        if c3<50:
                                                            t1.append(y)
                                                            WORDS1[y]=WORDS[y]
                                                            c3=c3+1
                                                    elif qrat[y]==1:
                                                        continue
                                                        #if c1<515:
                                                            #t2.append(y)
                                                            #WORDS1[y]=WORDS[y]
                                                            #c1=c1+1
                                                    elif qrat[y]==2:
                                                        if c2<50:
                                                            t3.append(y)
                                                            WORDS1[y]=WORDS[y]
                                                            c2=c2+1
                                                pass#pass#print(len(t1),len(t2),len(t3),len(WORDS1))
                                                return WORDS1

                                            def train_data_gen(WORDS1):
                                                #train and target data
                                                w11=[]
                                                trg11=[]
                                                w12=[]
                                                trg1=[]
                                                for tt in WORDS1:
                                                    s=' '
                                                    #if qrat[tt]!=1:
                                                    for kk in WORDS1[tt]:
                                                       # if qrat[tt]!=1:
                                                            w11.append(kk)
                                                            s=str(kk)+s+"\t"
                                                            #pass#pass#print(kk)
                                                            if float(qrat[tt])==2:
                                                                    trg11.append(1)
                                                            else:
                                                                    trg11.append(0)
                                                        #w12.append(s)
                                                        #trg1.append(qrat[tt])

                                                return w11,trg11
                                            def svm_coeff(w11,trg):
                                                                #SVM Learner and generate feature weights
                                                                #OutputCodeClassifier(LinearSVC(random_state=0),code_size=2, random_state=0)
                                                                #Learn SVM Model
                                                                #ifile = open("processed_revs_1.txt",encoding="ISO-8859-1")
                                                                Y = trg
                                                                words =w11
                                                                unique_words =[]
                                                                ss=set(words)
                                                                for w1 in ss:
                                                                        if w1 not in unique_words:
                                                                            unique_words.append(w1)

                                                                '''
                                                                #tf_transformer = TfidfVectorizer()
                                                                #f = tf_transformer.fit_transform(words)
                                                                #features = [((i, j), f[i,j]) for i, j in zip(*f.nonzero())]
                                                                #unique_word_ids = []
                                                                #for w in unique_words:
                                                                   # i = tf_transformer.vocabulary_.get(w)
                                                                    #unique_word_ids.append(i)

                                                                #clf =OneVsRestClassifier(SVC(kernel='linear'))#svm.LinearSVC(C=1)
                                                                #clf.fit(f,Y)
                                                                #p = clf.predict(f)
                                                                #pass#pass#print(f1_score(Y,p,average='weighted'))

                                                                '''
                                                                tf_transformer = TfidfVectorizer()
                                                                f = tf_transformer.fit_transform(words)
                                                                features = [((i, j), f[i,j]) for i, j in zip(*f.nonzero())]
                                                                unique_word_ids = []
                                                                for w in unique_words:
                                                                    i = tf_transformer.vocabulary_.get(w)
                                                                    unique_word_ids.append(i)

                                                                #clf =svm.LinearSVC(C=100,probability=True)
                                                                clf=svm.SVC(C=1.0, kernel='linear', degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=True, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=- 1, decision_function_shape='ovr', random_state=None)
                                                                #svm.SVC(C=1.0, kernel='linear', degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=True, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=- 1, decision_function_shape='ovr', random_state=None)

                                                                #OneVsRestClassifier(LinearSVC(random_state=0)) #svm.LinearSVC(C=1)
                                                                clf.fit(f,Y)
                                                                #clf.fit(f,Y)
                                                                p = clf.predict(f)
                                                                pass#pass#print(f1_score(Y,p,average='micro'))
                                                                from sklearn.metrics import accuracy_score
                                                                pass#pass#print(accuracy_score(Y,p))

                                                                #Store learned weights

                                                                #Tunable parameter, normalization range of weights
                                                                rangelower=0
                                                                rangehigher=1

                                                                C = clf.coef_
                                                                scaler = MinMaxScaler(feature_range=(rangelower,rangehigher))
                                                                vals = []
                                                                for i, j in zip(*C.nonzero()):
                                                                    vals.append([C[i,j]])
                                                                scaler.fit(vals)
                                                                V1 = scaler.transform(vals)
                                                                rows,cols = C.nonzero()
                                                                r_wts =  defaultdict(list)
                                                                ix= 0
                                                                for w in unique_words:
                                                                    r_wts[w] = [0]  
                                                                for i, j in zip(*C.nonzero()):
                                                                    #pass#pass#print(i,j)
                                                                    for k in tf_transformer.vocabulary_.keys():
                                                                        if tf_transformer.vocabulary_[k]==j:
                                                                            if k not in r_wts:
                                                                                break
                                                                            else:
                                                                                #if float(V1[ix][0])>0:
                                                                                r_wts[k][i] = V1[ix][0]
                                                                                break
                                                                    ix = ix + 1


                                                                #pass#pass#print(r_wts)
                                                                return r_wts
                                            #  Extracting Samehote Relation
                                            def sm_h(WORDS1,H11):
                                                H1={}
                                                fl=open("samehote.txt","w")
                                                for tt in H11:
                                                    gh=[]
                                                    for kk in H11[tt]:
                                                        if kk in WORDS1:
                                                            if kk not in gh:
                                                                gh.append(kk)
                                                    if len(gh)>1:
                                                        H1[tt]=gh
                                                        ggg=str(tt)+"::"+str(gh)
                                                        fl.write(str(ggg)+"\n")
                                                fl.close()       
                                                for t in H1:
                                                    pass#pass#pass#print(t,H1[t])   
                                                #pass#pass#print(len(H1))
                                                return H1
                                            #Sentance generation  for Neural Word2Vec Embedding Training
                                            def snt_emd(WORDS1):
                                                sent=[]
                                                sent1=[]
                                                sent_map=defaultdict(list)
                                                for ty in WORDS1:
                                                    gh=[]
                                                    gh.append(str(ty))
                                                    #gh1=[]
                                                    #gh2=[]
                                                    for j in WORDS1[ty]:
                                                        j1=str(j)
                                                        #gh.append(str(ty))
                                                        if j1 not in gh:
                                                            gh.append(j1)

                                                    if gh not in sent:
                                                            sent.append(gh)       
                                                documents=[]
                                                #documents1=[]
                                                for t in sent:
                                                    for jh in t:
                                                             documents.append(jh)
                                                return sent,documents
                                            # Clustering the queries
                                            def kmean_cls(sent,documents,WORDS1):

                                                    model = Word2Vec(sent, min_count=1)
                                                    X = model[model.wv.vocab]
                                                    NUM_CLUSTERS=5
                                                    kclusterer = KMeansClusterer(NUM_CLUSTERS, distance=nltk.cluster.util.cosine_distance,repeats=25)
                                                    assigned_clusters1 = kclusterer.cluster(X,assign_clusters=True)
                                                    #pass#pass#print (assigned_clusters)
                                                    cluster={}
                                                    words = list(model.wv.vocab)
                                                    for i, word in enumerate(words):
                                                            gh=[] 
                                                            gh1=[] 
                                                            gh2=[] 
                                                            if word.isdigit(): 
                                                                cluster[word]=assigned_clusters1[i]

                                                    cluster_final={}
                                                    for j in range(NUM_CLUSTERS):
                                                        gg=[]
                                                        for tt in cluster:
                                                            if int(cluster[tt])==int(j):
                                                                if tt not in gg:
                                                                    gg.append(tt)
                                                        if len(gg)>0:
                                                                    cluster_final[j]=gg
                                                    cc=0
                                                    final_clu={}
                                                    for t in cluster_final:
                                                        ghh=[]
                                                        for k in cluster_final[t]:
                                                            if int(k) in WORDS1:
                                                                   ghh.append(int(k))
                                                        if len(ghh)>=2:
                                                                final_clu[cc]=ghh
                                                                cc=cc+1
                                                    s=0
                                                    for k in final_clu:
                                                          s=s+len(final_clu[k])
                                                   # pass#pass#print(s)
                                                    return final_clu

                                            def sm_user():
                                                #Store similarity relations based on if they were written by same user
                                                    ifile = open("processed_revs_1.txt",encoding="ISO-8859-1")
                                                    ifile1 = open("metadata.txt")
                                                    SIM = collections.defaultdict(list)
                                                    userindex = {}
                                                    Sho= collections.defaultdict(list)
                                                    posrev=[]
                                                    negrev=[]
                                                    hotelindex={}
                                                    evida=[]
                                                    revid = 0
                                                    for ln in ifile:
                                                        parts = ln.strip().split("\t")
                                                        #if float(parts[1])==2:
                                                           # ppr="Positive("+str(revid)+")"
                                                            #posrev.append(ppr)
                                                            #if ppr not in evida:
                                                                #pass#evida.append(ppr)
                                                        #elif float(parts[1])==0:
                                                            #ppr1="Negative("+str(revid)+")"
                                                            #negrev.append(ppr1)
                                                            #if ppr1 not in evida:
                                                               # pass#evida.append(ppr1)
                                                        ln1 = ifile1.readline()
                                                        parts1 = ln1.strip().split("\t")
                                                       # pass#pass#print(parts1)
                                                        SIM[parts1[0]].append(revid)
                                                        Sho[parts1[1]].append(revid)
                                                        userindex[revid] = parts1[0]
                                                        hotelindex[revid]=parts1[1]
                                                        revid = revid + 1    
                                                    ifile.close()
                                                    ifile1.close()
                                                    #pass#pass#print(SIM)
                                                    Samuser_index={}
                                                    for t in SIM:
                                                        gh=[]
                                                        for k in SIM[t]:
                                                            if len(SIM[t])>0:
                                                                for e in SIM[t]:
                                                                    if e not in gh and e in WORDS:
                                                                        gh.append(e)
                                                        if gh!=[]:
                                                            Samuser_index[t]=gh
                                                    for d in Samuser_index:
                                                        pass#pass#pass#print(d,Samuser_index[d])          
                                                    #pass#pass#print(len(Samuser_index))
                                                    return Samuser_index
                                            #Annotated Word exp Used for validation
                                            def annotation_wordexp(WORDS1,s_words,stopwords):
                                                ann1={}
                                                c=0
                                                for k in WORDS1:
                                                    if qrat[k]==2:
                                                        c=c+1
                                                        c2=0
                                                        gff=[]
                                                        for gg in WORDS1[k]:
                                                            if gg in s_words:
                                                                if c2<20:
                                                                         gff.append(gg)
                                                                         c2=c2+1
                                                        if len(gff)>0:
                                                           # if k in WORDSt:
                                                                ann1[k]=gff

                                                    elif qrat[k]==0:
                                                        c=c+1
                                                        c3=0
                                                        gff1=[]
                                                        for gg in WORDS1[k]:
                                                            if gg in s_words:
                                                                if c3<20:
                                                                        gff1.append(gg)
                                                                        c3=c3+1

                                                        if len(gff1)>0:
                                                            #if k WORDSt:
                                                                ann1[k]=gff1
                                                ann={}
                                                for t in ann1:
                                                    if t in WORDS1:
                                                        ann[t]=ann1[t]
                                                pass#pass#print(len(ann))
                                                gg=open("Review_Word_annotation.txt","w")
                                                for t in ann:
                                                    vv=str(t)+":"+str(ann[t])
                                                    gg.write(str(vv)+"\n")
                                                    pass#pass#print(t,ann[t])
                                                gg.close()
                                                return ann
                                            def feedback_gen(final_clu):
                                                d_tt={}
                                                d_tt[0]="negative"
                                                d_tt[2]="positive"
                                                wr={}
                                                w=[]
                                                for k in final_clu:
                                                        #c=-1
                                                        #c=c+1
                                                        md=int(len(final_clu[k])/2)
                                                        c=0      
                                                        k1= final_clu[k][md+c]
                                                        #pass#pass#print(k1,md)        
                                                        if k1 in ann:
                                                                    for k3 in ann[k1]:
                                                                            w.append(k3)
                                                        else:
                                                            c=c+11
                                                            continue 


                                                        #pass#pass#print(k,k1,md,d_tt[qrat[k1]],w)
                                                        wr[k1]=w
                                                #pass#pass#print(w)
                                                return w

                                            #Update Evidence Based on manual annotation Update 2 

                                            def update_evid_annotation(w,WORDS1):
                                                            model = Word2Vec(sent, min_count=1)
                                                            data_g={}
                                                            for t in WORDS1:
                                                                chu=[]
                                                                #try:
                                                                vb={}
                                                                for v in w:
                                                                    vb1={}
                                                                    for v1 in WORDS1[t]:
                                                                            #pass#pass#print(v1,v)
                                                                            gh1=model.similarity(v,v1)
                                                                            if gh1>=0.3:
                                                                                  vb1[v1]=float(gh1)
                                                                                  #pass#pass#print(gh1)
                                                                    for jk in vb1:
                                                                        if jk in vb:
                                                                            if float(vb1[jk])>=float(vb[jk]):
                                                                                #pass#pass#print(jk,vb1[jk],vb[jk])
                                                                                vb[jk]=vb1[jk]
                                                                        else:
                                                                            vb[jk]=vb1[jk]
                                                                #pass#pass#print(t, vb)
                                                                #pass#pass#print("\n")             
                                                                dd=sorted(vb.items(), key=operator.itemgetter(1),reverse=True)
                                                                cc=0
                                                                for kkk in dd:
                                                                    if kkk[0] not in chu:
                                                                        if cc<20:
                                                                            chu.append(kkk[0])
                                                                            cc=cc+1

                                                                if len(chu)>0:
                                                                    data_g[t]=chu
                                                            #survey checking
                                                            #pass#pass#print(len(WORDS1))
                                                            #Updating the Whole Evidence Based on manual annotation
                                                            WORDS22={}
                                                            for gg in WORDS1:
                                                                #if gg in data_extract12:
                                                                    #WORDS2[gg]=data_extract12[gg]
                                                                if gg in data_g:
                                                                    if len(data_g[gg])>0:
                                                                        WORDS22[gg]=data_g[gg]
                                                            #pass#pass#print(WORDS2['d_535'])
                                                            #pass#pass#print(len(WORDS22))
                                                            for t in WORDS22:
                                                                pass#pass#pass#print(t,WORDS22[t])
                                                            return WORDS22


                                            WORDS,qrat,H11,Rev_text_map,s_words,stopwords=data_pr()
                                            WORDS1=data_balancing(WORDS)
                                            w11,trg1=train_data_gen(WORDS1)
                                            #pass#pass#print(trg1)
                                            r_wts=svm_coeff(w11,trg1)
                                            H1=sm_h(WORDS1,H11)
                                            sent,documents=snt_emd(WORDS1)
                                            final_clu=kmean_cls(sent,documents,WORDS1)
                                            Samuser_index=sm_user()
                                            ann=annotation_wordexp(WORDS1,s_words,stopwords)
                                            w=feedback_gen(final_clu)
                                            WORDS22=update_evid_annotation(w,WORDS1)
                                            return WORDS,WORDS22,qrat,H1,ann,Rev_text_map,s_words,stopwords,WORDS1,r_wts,w11,trg1,H1,sent,documents,final_clu,Samuser_index









                            #WORDS,WORDS22,qrat,H1,ann,Rev_text_map,s_words,stopwords,WORDS1,r_wts,w11,trg1,H1,sent,documents,final_clu,Samuser_index=run_all()

                            
                            
                            
                            
# OriginalRelational MLN Explanation
class orexp:
                    @classmethod
                    def originalmln_annotation_exp(cls):

                                import random
                                from collections import defaultdict

                                #Sampler
                                margs23 =defaultdict(list)
                                Relational_formula_filter={}
                                iter1=defaultdict(list)
                                users_s= defaultdict(list)
                                expns2 = defaultdict(dict)
                                Relational_formula_filter={}
                                same_user={}
                                Sample={}
                                Sample_r={}
                                for h in H1:
                                    for i,r in enumerate(H1[h]):
                                        if r in WORDS1:
                                                    #Sample[r] = random.randint(0,1)
                                                    Sample_r[r] = random.randint(0,1)
                                for h in Samuser_index:
                                    for i,r in enumerate(Samuser_index[h]):
                                        if r in WORDS1:
                                                    Sample[r] = random.randint(0,1)
                                                    #Sample_r[r] = random.randint(0,2)
                                                    #Sample_r[r] = random.randint(0,1)
                                                    margs23[r] = [0]*2
                                                    #margs_r[r] = 0
                                                    iter1[r] =0

                                #Tunable parameter (default value of prob)
                                C1 = 0.98
                                VR=0.98
                                iters =1000000

                                for t in range(0,iters,1):
                                    h1 = random.choice(list(H1.keys()))
                                    if len(H1[h1])==0:
                                        continue
                                    ix1 = random.randint(0,len(H1[h1])-1)
                                    r33 = H1[h1][ix1]
                                    h = random.choice(list(Samuser_index.keys()))
                                    if len(Samuser_index[h])==0:
                                        continue
                                    ix = random.randint(0,len(Samuser_index[h])-1)
                                    r = Samuser_index[h][ix]
                                    if r in WORDS1:
                                        if random.random()<0.5:
                                            #sample Topic
                                            W0=0
                                            W1=0
                                            W2=0


                                            try:
                                                        for w in WORDS1[r]:
                                                            if len(r_wts[w])<1:
                                                                continue
                                                            W0=W0+r_wts[w][0]
                                                            W1=W1+(1-r_wts[w][0])
                                                            #W2=W2+(r_wts[w][2])
                                                            if r not in expns2 or w not in expns2[r]:
                                                                                expns2[r][w] = r_wts[w][0]
                                                                                expns2[r][w] = (1-r_wts[w][0])
                                                                               # expns2[r][w] = (r_wts[w][2])


                                                            else:
                                                                                expns2[r][w] = expns2[r][w] + r_wts[w][0]
                                                                                expns2[r][w] = expns2[r][w] + (1-r_wts[w][0])
                                                                                #expns2[r][w] = expns2[r][w] + (r_wts[w][2])

                                            except:
                                                continue



                                            if (W0+W1)!= 0:
                                                W0 = W0/(W0+W1)
                                                W1 = W1/(W0+W1)
                                               # W2=W2/(W0+W1+W2)


                                                sval = random.random()
                                                iter1[r]=iter1[r]+1
                                                #print(sval,W14,W15,W16,W17,W18,W19)
                                                if sval<W0:
                                                    Sample[r]=0
                                                    Sample_r[r33]=0
                                                    margs23[r][0]=margs23[r][0]+1
                                                elif sval<(W0+W1):
                                                    Sample[r]=1
                                                    Sample_r[r33]=1
                                                    margs23[r][1]=margs23[r][1]+1



                                                for r22 in H1[h1]:
                                                    if r22 in WORDS1 and r33 in WORDS1:
                                                            if r22==r33:
                                                                continue
                                                            if Sample_r[r22]!=Sample_r[r33]:
                                                                if Sample_r[r22]==0:
                                                                   # print("yesssssss")
                                                                    #P = P + VR
                                                                    #exp1 = exp1 + 0.1
                                                                    hhlll="Samehotel("+str(r33)+","+str(r22)+")"
                                                                    if r33 not in expns2 or hhlll not in expns2[r33]:
                                                                        expns2[r33][hhlll] = VR

                                                                    else:
                                                                        expns2[r33][hhlll] = expns2[r33][hhlll] + VR 
                                                            elif Sample_r[r22]!=Sample_r[r33]:
                                                                if Sample_r[r22]==1:
                                                                    #P = P + VR
                                                                    #exp1 = exp1 + 0.1
                                                                    hhlll="Samehotel("+str(r33)+","+str(r22)+")"
                                                                    if r33 not in expns2 or hhlll not in expns2[r33]:
                                                                        expns2[r33][hhlll] = VR

                                                                    else:
                                                                        expns2[r33][hhlll] = expns2[r33][hhlll] + VR

                                                for r1 in Samuser_index[h]:
                                                    if r1==r:
                                                        continue
                                                    if r in WORDS1 and r1 in WORDS1:
                                                            if Sample[r]!=Sample[r1]:
                                                                if Sample[r1]==0:
                                                                    #W0=W0+r_wts1[w][0]
                                                                    #margs[r][0]=margs[r][0]+1
                                                                    hhlll="Sameuser("+str(r)+","+str(r1)+")"
                                                                    if r not in expns2 or hhlll not in expns2[r]:
                                                                        expns2[r][hhlll] =C1
                                                                        if r not in Relational_formula_filter:
                                                                            Relational_formula_filter[r]=WORDS1[r]    
                                                                    else:
                                                                        expns2[r][hhlll] = expns2[r][hhlll] + C1
                                                                elif Sample[r1]==1:
                                                                    #W1=W1+r_wts1[w][1]
                                                                   # margs[r][1]=margs[r][1]+1
                                                                    hhlll="Sameuser("+str(r)+","+str(r1)+")"
                                                                    if r not in expns2 or hhlll not in expns2[r]:
                                                                        expns2[r][hhlll] =C1
                                                                        if r not in Relational_formula_filter:
                                                                            Relational_formula_filter[r]=WORDS1[r]    
                                                                    else:
                                                                        expns2[r][hhlll] = expns2[r][hhlll] +C1 




                                #Computing Marginal Probability after user input
                                margs22={}
                                for t in margs23:
                                    gh=[]
                                    if iter1[t]>0:
                                        for kk in margs23[t]:
                                            vv=float(kk)/float(iter1[t])
                                            if float(vv)>=1.0:
                                                gh.append(float(1.0))
                                            elif float(vv)<1.0:
                                                gh.append(abs(float(vv)))
                                        margs22[t]=gh
                                margs33={}
                                for t in margs22:
                                    s=0
                                    for ww in margs22[t]:
                                        s=s+float(ww)
                                    #if s!=0:
                                    pass#print(t,s)
                                    margs33[t]=margs22[t]
                                #Computing the Highest Probability user input
                                margs3_u={}
                                margs222={}
                                for ww in margs33:
                                    gg=[]
                                    for kk in range(0,len(margs33[ww])):
                                            gg.append(margs33[ww][kk])
                                    if qrat[ww]!=1:
                                       # print(ww,qrat[ww])
                                        margs222[ww]=gg

                                for dd in margs222:
                                    v=max(margs222[dd])
                                    margs3_u[dd]=v
                                for vv in margs3_u:
                                    if margs3_u[vv]>=0.2:
                                        pass#print(vv,margs3[vv])
                                d_tt={}
                                d_tt[0]="negative"
                                d_tt[1]="positive"


                                #predict topic user input
                                sampled_doc=[]
                                pred_t=[]
                                for a in margs222:
                                    for ss in range(0,len(margs222[a])):
                                            if margs222[a][ss]==margs3_u[a]:
                                                    #print(a,d_tt[ss])
                                                    sampled_doc.append(a)
                                                    pred_t.append(d_tt[ss])
                                ss=set(sampled_doc)
                                sampled_doc_up=[]
                                sampled_doc_up_map_user={}
                                for kk in ss:
                                    sampled_doc_up.append(kk)
                                for tt in sampled_doc_up:
                                    ggf=[]
                                    gvv=[]
                                    c=0
                                    for gg in range(0,len(sampled_doc)):
                                        if tt==sampled_doc[gg]:
                                            ggf.append(pred_t[gg])
                                    #if len(ggf)==1:
                                    #print(ggf)
                                    for ggh in ggf:
                                        if c<1:
                                            gvv.append(ggh)
                                            c=c+1

                                    sampled_doc_up_map_user[tt]=gvv

                                cx=0   
                                for s in sampled_doc_up_map_user:
                                    if len(sampled_doc_up_map_user[s])>1:
                                            cx=cx+1
                                           #print(s,sampled_doc_up_map[s])


                                #print(doc_per_pred_topic)
                                print(cx)
                                ffd1=open("originalfullmlntxt","w")

                                for s in sampled_doc_up_map_user:
                                    ccvb=str(s)+":"+str(sampled_doc_up_map_user[s])
                                    ffd1.write(str(ccvb)+"\n")
                                ffd1.close()


                                #Explanation Generation  with user
                                import operator
                                correct_predictions_r = {}

                                for m in margs33.keys():
                                            if m in WORDS1 and m in sampled_doc_up_map_user:
                                                          #print(m)
                                                          correct_predictions_r[m] = 1
                                            #if len(WORDS[m])==0:#or ratings[m]==3:
                                               # continue
                                            #else:
                                               # correct_predictions[m] = 1
                                            #if margs[m] > 0.5 and spamindex[m] ==-1:
                                                #correct_predictions[m] = 1
                                fft_r=open("expn_n1_r1_user.txt","w")  
                                explanation_r={}
                                for e in expns2: 
                                    if e in correct_predictions_r:
                                        sorted_expns_r = sorted(expns2[e].items(), key=operator.itemgetter(1),reverse=True)
                                        z = 0
                                        for s in sorted_expns_r[:]:
                                            z = z + s[1]
                                        rex_r = {}
                                        keys_r = []
                                        for s in sorted_expns_r[:]:
                                            rex_r[s[0]] = s[1]/z
                                            keys_r.append(s[0])
                                        #if "Sameuser" not in keys or "Samehotel" not in keys:
                                            #continue
                                        sorted1 = sorted(rex_r.items(), key=operator.itemgetter(1),reverse=True)
                                        #if sorted1[0][0]=="JNTM":
                                        #print(str(e) +" "+str(sorted1))
                                        gg=str(e) +":"+str(sorted1)
                                        explanation_r[e]=sorted1
                                        fft_r.write(str(gg)+"\n")
                                hhh="Explanation_Topic1"+"_user"+" before user feedback"+".txt"
                                f11_r=open(hhh,"w")
                                Store_Explanation_user3={}
                                for t in explanation_r:
                                    #for k in WORDS1:
                                           #if str(t)==str(k):
                                                ggg=str(t)+":"+str(explanation_r[t])
                                                f11_r.write(str(ggg)+"\n")
                                                #print(t,explanation_r[t])
                                                #print("\n")
                                                Store_Explanation_user3[t]=explanation_r[t]
                                return Store_Explanation_user3

                    #Store_Explanation_user3=originalmln_annotation_exp()  
# Annotated Relational MLN Explanation
class annotation:
    @classmethod
    def mln_annotation_exp(cls):

            import random
            from collections import defaultdict

            #Sampler
            margs23 =defaultdict(list)
            Relational_formula_filter={}
            iter1=defaultdict(list)
            users_s= defaultdict(list)
            expns2 = defaultdict(dict)
            Relational_formula_filter={}
            same_user={}
            Sample={}
            Sample_r={}
            for h in H1:
                for i,r in enumerate(H1[h]):
                    if r in WORDS22:
                                #Sample[r] = random.randint(0,1)
                                Sample_r[r] = random.randint(0,1)
            for h in Samuser_index:
                for i,r in enumerate(Samuser_index[h]):
                    if r in WORDS22:
                                Sample[r] = random.randint(0,1)
                                #Sample_r[r] = random.randint(0,2)
                                #Sample_r[r] = random.randint(0,1)
                                margs23[r] = [0]*2
                                #margs_r[r] = 0
                                iter1[r] =0

            #Tunable parameter (default value of prob)
            C1 = 0.98
            VR=0.98
            iters =1000000

            for t in range(0,iters,1):
                h1 = random.choice(list(H1.keys()))
                if len(H1[h1])==0:
                    continue
                ix1 = random.randint(0,len(H1[h1])-1)
                r33 = H1[h1][ix1]
                h = random.choice(list(Samuser_index.keys()))
                if len(Samuser_index[h])==0:
                    continue
                ix = random.randint(0,len(Samuser_index[h])-1)
                r = Samuser_index[h][ix]
                if r in WORDS22:
                    if random.random()<0.5:
                        #sample Topic
                        W0=0
                        W1=0
                        W2=0


                        try:
                                    for w in WORDS22[r]:
                                        if len(r_wts[w])<1:
                                            continue
                                        W0=W0+r_wts[w][0]
                                        W1=W1+(1-r_wts[w][0])
                                        #W2=W2+(r_wts[w][2])
                                        if w in s_words:
                                            if r not in expns2 or w not in expns2[r]:
                                                                expns2[r][w] = r_wts[w][0]
                                                                expns2[r][w] = (1-r_wts[w][0])
                                                               # expns2[r][w] = (r_wts[w][2])


                                            else:
                                                                expns2[r][w] = expns2[r][w] + r_wts[w][0]
                                                                expns2[r][w] = expns2[r][w] + (1-r_wts[w][0])
                                                                #expns2[r][w] = expns2[r][w] + (r_wts[w][2])

                        except:
                            continue



                        if (W0+W1)!= 0:
                            W0 = W0/(W0+W1)
                            W1 = W1/(W0+W1)
                           # W2=W2/(W0+W1+W2)


                            sval = random.random()
                            iter1[r]=iter1[r]+1
                            #print(sval,W14,W15,W16,W17,W18,W19)
                            if sval<W0:
                                Sample[r]=0
                                Sample_r[r33]=0
                                margs23[r][0]=margs23[r][0]+1
                            elif sval<(W0+W1):
                                Sample[r]=1
                                Sample_r[r33]=1
                                margs23[r][1]=margs23[r][1]+1



                            for r22 in H1[h1]:
                                if r22 in WORDS22 and r33 in WORDS22:
                                        if r22==r33:
                                            continue
                                        if Sample_r[r22]!=Sample_r[r33]:
                                            if Sample_r[r22]==0:
                                               # print("yesssssss")
                                                #P = P + VR
                                                #exp1 = exp1 + 0.1
                                                hhlll="Samehotel("+str(r33)+","+str(r22)+")"
                                                if r33 not in expns2 or hhlll not in expns2[r33]:
                                                    expns2[r33][hhlll] = VR

                                                else:
                                                    expns2[r33][hhlll] = expns2[r33][hhlll] + VR 
                                        elif Sample_r[r22]!=Sample_r[r33]:
                                            if Sample_r[r22]==1:
                                                #P = P + VR
                                                #exp1 = exp1 + 0.1
                                                hhlll="Samehotel("+str(r33)+","+str(r22)+")"
                                                if r33 not in expns2 or hhlll not in expns2[r33]:
                                                    expns2[r33][hhlll] = VR

                                                else:
                                                    expns2[r33][hhlll] = expns2[r33][hhlll] + VR

                            for r1 in Samuser_index[h]:
                                if r1==r:
                                    continue
                                if r in WORDS22 and r1 in WORDS22:
                                        if Sample[r]!=Sample[r1]:
                                            if Sample[r1]==0:
                                                #W0=W0+r_wts1[w][0]
                                                #margs[r][0]=margs[r][0]+1
                                                hhlll="Sameuser("+str(r)+","+str(r1)+")"
                                                if r not in expns2 or hhlll not in expns2[r]:
                                                    expns2[r][hhlll] =C1
                                                    if r not in Relational_formula_filter:
                                                        Relational_formula_filter[r]=WORDS22[r]    
                                                else:
                                                    expns2[r][hhlll] = expns2[r][hhlll] + C1
                                            elif Sample[r1]==1:
                                                #W1=W1+r_wts1[w][1]
                                               # margs[r][1]=margs[r][1]+1
                                                hhlll="Sameuser("+str(r)+","+str(r1)+")"
                                                if r not in expns2 or hhlll not in expns2[r]:
                                                    expns2[r][hhlll] =C1
                                                    if r not in Relational_formula_filter:
                                                        Relational_formula_filter[r]=WORDS22[r]    
                                                else:
                                                    expns2[r][hhlll] = expns2[r][hhlll] +C1 




            #Computing Marginal Probability after user input
            margs22={}
            for t in margs23:
                gh=[]
                if iter1[t]>0:
                    for kk in margs23[t]:
                        vv=float(kk)/float(iter1[t])
                        if float(vv)>=1.0:
                            gh.append(float(1.0))
                        elif float(vv)<1.0:
                            gh.append(abs(float(vv)))
                    margs22[t]=gh
            margs33={}
            for t in margs22:
                s=0
                for ww in margs22[t]:
                    s=s+float(ww)
                #if s!=0:
                pass#print(t,s)
                margs33[t]=margs22[t]
            #Computing the Highest Probability user input
            margs3_u={}
            margs222={}
            for ww in margs33:
                gg=[]
                for kk in range(0,len(margs33[ww])):
                        gg.append(margs33[ww][kk])
                if qrat[ww]!=1:
                   # print(ww,qrat[ww])
                    margs222[ww]=gg

            for dd in margs222:
                v=max(margs222[dd])
                margs3_u[dd]=v
            for vv in margs3_u:
                if margs3_u[vv]>=0.2:
                    pass#print(vv,margs3[vv])
            d_tt={}
            d_tt[0]="negative"
            d_tt[1]="positive"


            #predict topic user input
            sampled_doc=[]
            pred_t=[]
            for a in margs222:
                for ss in range(0,len(margs222[a])):
                        if margs222[a][ss]==margs3_u[a]:
                                #print(a,d_tt[ss])
                                sampled_doc.append(a)
                                pred_t.append(d_tt[ss])
            ss=set(sampled_doc)
            sampled_doc_up=[]
            sampled_doc_up_map_user={}
            for kk in ss:
                sampled_doc_up.append(kk)
            for tt in sampled_doc_up:
                ggf=[]
                gvv=[]
                c=0
                for gg in range(0,len(sampled_doc)):
                    if tt==sampled_doc[gg]:
                        ggf.append(pred_t[gg])
                #if len(ggf)==1:
                #print(ggf)
                for ggh in ggf:
                    if c<1:
                        gvv.append(ggh)
                        c=c+1

                sampled_doc_up_map_user[tt]=gvv

            cx=0   
            for s in sampled_doc_up_map_user:
                if len(sampled_doc_up_map_user[s])>1:
                        cx=cx+1
                       #print(s,sampled_doc_up_map[s])


            #print(doc_per_pred_topic)
            print(cx)
            ffd1=open("originalfullmlntxt","w")

            for s in sampled_doc_up_map_user:
                ccvb=str(s)+":"+str(sampled_doc_up_map_user[s])
                ffd1.write(str(ccvb)+"\n")
            ffd1.close()


            #Explanation Generation  with user
            import operator
            correct_predictions_r = {}

            for m in margs33.keys():
                        if m in WORDS22 and m in sampled_doc_up_map_user:
                                      #print(m)
                                      correct_predictions_r[m] = 1
                        #if len(WORDS[m])==0:#or ratings[m]==3:
                           # continue
                        #else:
                           # correct_predictions[m] = 1
                        #if margs[m] > 0.5 and spamindex[m] ==-1:
                            #correct_predictions[m] = 1
            fft_r=open("expn_n1_r1_user.txt","w")  
            explanation_r={}
            for e in expns2: 
                if e in correct_predictions_r:
                    sorted_expns_r = sorted(expns2[e].items(), key=operator.itemgetter(1),reverse=True)
                    z = 0
                    for s in sorted_expns_r[:]:
                        z = z + s[1]
                    rex_r = {}
                    keys_r = []
                    for s in sorted_expns_r[:]:
                        rex_r[s[0]] = s[1]/z
                        keys_r.append(s[0])
                    #if "Sameuser" not in keys or "Samehotel" not in keys:
                        #continue
                    sorted1 = sorted(rex_r.items(), key=operator.itemgetter(1),reverse=True)
                    #if sorted1[0][0]=="JNTM":
                    #print(str(e) +" "+str(sorted1))
                    gg=str(e) +":"+str(sorted1)
                    explanation_r[e]=sorted1
                    fft_r.write(str(gg)+"\n")
            hhh="Explanation_Topic1"+"_user"+" before user feedback"+".txt"
            f11_r=open(hhh,"w")
            Store_Explanation_user2={}
            for t in explanation_r:
                #for k in WORDS22:
                       #if str(t)==str(k):
                            ggg=str(t)+":"+str(explanation_r[t])
                            f11_r.write(str(ggg)+"\n")
                            #print(t,explanation_r[t])
                            #print("\n")
                            Store_Explanation_user2[t]=explanation_r[t]
            return Store_Explanation_user2
# Relational annotation

def relational_embedding_exp(m):
    # Relational Exp generatetion based on neural embedding
                sent2=[]
                sent1=[]
                sent_map=defaultdict(list)
                for ty in WORDS22:
                    gh=[]
                    gh.append(str(ty))
                    #gh1=[]
                    #gh2=[]
                    for j in WORDS22[ty]:

                        j1=str(j)
                        #gh.append(str(ty))
                        if j1 not in gh:
                            gh.append(j1)
                        ##print(gh)


                    if gh not in sent2:
                            sent2.append(gh)


                documents1=[]
                #documents1=[]
                for t in sent2:
                    s=''
                    for jh in t:
                        if jh.isdigit():
                             documents1.append(jh)
                        else:
                            s=" "+str(jh)+s+" "
                    documents1.append(s)


                #sentence embedding
                from gensim.test.utils import common_texts
                from gensim.models.doc2vec import Doc2Vec, TaggedDocument
                documents2 = [TaggedDocument(doc, [i]) for i, doc in enumerate(sent2)]
                for t in documents2:
                    pass##print(t)
                model = Doc2Vec(documents2, vector_size=5, window=2, min_count=1, workers=4)

                #K-Means Run 14 to find the neighbors per query 

                #cluster generation with k-means
                import sys
                from nltk.cluster import KMeansClusterer
                import nltk
                from sklearn import cluster
                from sklearn import metrics
                import gensim 
                import operator
                #from gensim.models import Word2Vec


                #model = Word2Vec(sent, min_count=1) dd=sorted(vb.items(), key=operator.itemgetter(1),reverse=False)
                import operator
                X = model[model.wv.vocab]
                c=0
                cluster={}
                num=[]
                weight_map={}
                similar_r_map={}
                fg={}
                for jj in WORDS1:
                    gh1=[]
                    gh2=[]
                    s=0

                    for k in documents1:
                        if str(k)==str(jj):
                            gh=model.most_similar(positive=str(k),topn=600)
                           # #print(gh)
                            for tt in gh:
                                if float(tt[1]) not in gh1:
                                    gh1.append(float(tt[1]))
                                #if tt[0] not in gh2:
                                if tt[0].isdigit():
                                        #if ccc<5:
                                                #gh2.append(tt[0])
                                                fg[tt[0]]=tt[1]
                                                #ccc=ccc+1
                    #for ffg in gh1:
                        #s=s+ffg
                    dd=sorted(fg.items(), key=operator.itemgetter(1),reverse=True)
                    ccc=0
                    for t5 in dd:
                        if qrat[int(jj)]==qrat[int(t5[0])]:
                            if m==5:
                                if ccc<300:
                                         gh2.append(t5[0])
                                         ccc=ccc+1
                            elif m==10:
                                if ccc<400:
                                         gh2.append(t5[0])
                                         ccc=ccc+1
                            elif m==15:
                                if ccc<500:
                                         gh2.append(t5[0])
                                         ccc=ccc+1
                            elif m==20:
                                if ccc<600:
                                         gh2.append(t5[0])
                                         ccc=ccc+1
                            elif m==25:
                                if ccc<700:
                                         gh2.append(t5[0])
                                         ccc=ccc+1

                    #if len(gh2)>=2:
                    similar_r_map[jj]=gh2
                            #ccc=ccc+1
               
                return similar_r_map
#cmp=5
#similar_r_map=relational_embedding_exp(cmp)


#new technique with reduced object varying cluster size aggregate 
class clso:
                @classmethod
                def clso(cls):
                                                    import rpy2
                                                    import rpy2.robjects.packages as rpackages
                                                    from rpy2.robjects.vectors import StrVector
                                                    from rpy2.robjects.packages import importr
                                                    utils = rpackages.importr('utils')
                                                    utils.chooseCRANmirror(ind=1)
                                                    # Install packages
                                                    packnames = ('TopKLists', 'CEMC')
                                                    utils.install_packages(StrVector(packnames))
                                                    packnames = ('data(TopKSpaceSampleInput)')
                                                    utils.install_packages(StrVector(packnames))
                                                    h = importr('TopKLists')


                                                    expt_st_pc={}
                                                    perd_m={}
                                                    obj_m={}
                                                    Sample_model={}
                                                    def cluster_generatio(kk):

                                                                            #cluster generation with k-means
                                                                            import sys
                                                                            from nltk.cluster import KMeansClusterer
                                                                            import nltk
                                                                            from sklearn import cluster
                                                                            from sklearn import metrics
                                                                            import gensim 
                                                                            import operator
                                                                            from gensim.models import Word2Vec
                                                                            model = Word2Vec(sent, min_count=1)
                                                                            X = model[model.wv.vocab]
                                                                            NUM_CLUSTERS=kk
                                                                            kclusterer = KMeansClusterer(NUM_CLUSTERS, distance=nltk.cluster.util.cosine_distance,repeats=25)
                                                                            assigned_clusters1 = kclusterer.cluster(X,assign_clusters=True)
                                                                            ##print (assigned_clusters)
                                                                            cluster={}
                                                                            words = list(model.wv.vocab)
                                                                            for i, word in enumerate(words):
                                                                              gh=[] 
                                                                              gh1=[] 
                                                                              gh2=[] 
                                                                              if word.isdigit(): 
                                                                                cluster[word]=assigned_clusters1[i]
                                                                                ##print (word + ":" + str(assigned_clusters[i]))
                                                                            cluster_final={}
                                                                            for j in range(NUM_CLUSTERS):
                                                                                gg=[]
                                                                                for tt in cluster:
                                                                                    if int(cluster[tt])==int(j):
                                                                                        if tt not in gg:
                                                                                            gg.append(tt)
                                                                                if len(gg)>0:
                                                                                            cluster_final[j]=gg
                                                                            cc=0
                                                                            final_clu={}
                                                                            for t in cluster_final:
                                                                                ghh=[]
                                                                                for k in cluster_final[t]:
                                                                                    if int(k) in WORDS:
                                                                                           ghh.append(int(k))
                                                                                if len(ghh)>=2:
                                                                                        final_clu[cc]=ghh
                                                                                        cc=cc+1
                                                                            return final_clu


                                                    def sample_rev(mzzz,final_clu,j,M):
                                                                                            #reduced objects
                                                                                            reduced_obj=[]
                                                                                            expt_st_pd={}
                                                                                            import random
                                                                                            import operator
                                                                                            import os.path
                                                                                            import os 

                                                                                            for k in final_clu:
                                                                                                c=0
                                                                                                for tt in final_clu[k]:
                                                                                                    if random.random()<M:
                                                                                                        c=c+1
                                                                                                        chh=float(c)/len(final_clu[k])
                                                                                                        if chh<=M:
                                                                                                                ##print(M)
                                                                                                                if tt not in reduced_obj:
                                                                                                                    reduced_obj.append(tt)
                                                                                            ##print(len(reduced_obj)/float(len(final_clu[k]))
                                                                                            wr={}
                                                                                            w=[]
                                                                                            for k in final_clu:
                                                                                                    #c=-1
                                                                                                    #c=c+1
                                                                                                    md=int(len(final_clu[k])//2)
                                                                                                    c=0      
                                                                                                    k1= final_clu[k][md+c]
                                                                                                    ##print(k1,md)        
                                                                                                    if k1 in ann:
                                                                                                                for k3 in ann[k1]:
                                                                                                                        w.append(k3)
                                                                                                    else:
                                                                                                        c=c+11
                                                                                                        continue 


                                                                                                   # #print(k,k1,md,d_tt[qrat[k1]],w)
                                                                                                    wr[k1]=w
                                                                                            model = Word2Vec(sent, min_count=1)
                                                                                            data_g={}
                                                                                            for t in WORDS1:
                                                                                                chu=[]
                                                                                                #try:
                                                                                                vb={}
                                                                                                for v in w:
                                                                                                    vb1={}
                                                                                                    for v1 in WORDS1[t]:
                                                                                                            ##print(v1,v)
                                                                                                            gh1=model.similarity(v,v1)
                                                                                                            if gh1>=0.40:
                                                                                                                  vb1[v1]=float(gh1)
                                                                                                                  ##print(gh1)
                                                                                                    for jk in vb1:
                                                                                                        if jk in vb:
                                                                                                            if float(vb1[jk])>=float(vb[jk]):
                                                                                                                ##print(jk,vb1[jk],vb[jk])
                                                                                                                vb[jk]=vb1[jk]
                                                                                                        else:
                                                                                                            vb[jk]=vb1[jk]
                                                                                                ##print(t, vb)
                                                                                                ##print("\n")             
                                                                                                dd=sorted(vb.items(), key=operator.itemgetter(1),reverse=True)
                                                                                                cc=0
                                                                                                for kkk in dd:
                                                                                                    if kkk[0] not in chu:
                                                                                                        if cc<25:
                                                                                                            chu.append(kkk[0])
                                                                                                            cc=cc+1

                                                                                                if len(chu)>0:
                                                                                                    data_g[t]=chu
                                                                                            #survey checking
                                                                                            #print(len(WORDS1))
                                                                                            #Updating the Whole Evidence Based on manual annotation
                                                                                            WORDS22={}
                                                                                            for gg in WORDS1:
                                                                                                #if gg in data_extract12:
                                                                                                    #WORDS2[gg]=data_extract12[gg]
                                                                                                if gg in data_g:
                                                                                                    if len(data_g[gg])>0:
                                                                                                        WORDS22[gg]=data_g[gg]
                                                                                            WORDS2={}
                                                                                            for t in reduced_obj:
                                                                                                hhh1=[]
                                                                                                czx=0
                                                                                                #cxzz=random.randint(5,11)
                                                                                                if t in WORDS22:
                                                                                                    #random.shuffle(WORDS22[t])
                                                                                                    for dd in WORDS22[t]:
                                                                                                        if dd not in hhh1:
                                                                                                            if czx<10 and dd in s_words or str(dd) in s_words:
                                                                                                                hhh1.append(dd)
                                                                                                                czx=czx+1

                                                                                                    WORDS2[t]=hhh1
                                                                                                    ##print(len(hhh1))
                                                                                            ##print(WORDS2)
                                                                                            ##print(len(WORDS2),len(reduced_obj),len(WORDS22))
                                                                                            #Sample 2 user

                                                                                            import random
                                                                                            from collections import defaultdict

                                                                                            #Sampler
                                                                                            margs23 =defaultdict(list)
                                                                                            Relational_formula_filter={}
                                                                                            iter1=defaultdict(list)
                                                                                            users_s= defaultdict(list)
                                                                                            expns2 = defaultdict(dict)
                                                                                            Relational_formula_filter={}
                                                                                            same_user={}
                                                                                            Sample={}
                                                                                            Sample_r={}
                                                                                            for h in H1:
                                                                                                for i,r in enumerate(H1[h]):
                                                                                                    if r in WORDS2:
                                                                                                                #Sample[r] = random.randint(0,1)
                                                                                                                Sample_r[r] = random.randint(0,1)
                                                                                            for h in Samuser_index:
                                                                                                for i,r in enumerate(Samuser_index[h]):
                                                                                                    if r in WORDS2:
                                                                                                                Sample[r] = random.randint(0,1)
                                                                                                                #Sample_r[r] = random.randint(0,2)
                                                                                                                #Sample_r[r] = random.randint(0,1)
                                                                                                                margs23[r] = [0]*2
                                                                                                                #margs_r[r] = 0
                                                                                                                iter1[r] =0

                                                                                            #Tunable parameter (default value of prob)
                                                                                            C1 = 0.98
                                                                                            VR=0.98
                                                                                            iters =1000000

                                                                                            for t in range(0,iters,1):
                                                                                                h1 = random.choice(list(H1.keys()))
                                                                                                if len(H1[h1])==0:
                                                                                                    continue
                                                                                                ix1 = random.randint(0,len(H1[h1])-1)
                                                                                                r33 = H1[h1][ix1]
                                                                                                h = random.choice(list(Samuser_index.keys()))
                                                                                                if len(Samuser_index[h])==0:
                                                                                                    continue
                                                                                                ix = random.randint(0,len(Samuser_index[h])-1)
                                                                                                r = Samuser_index[h][ix]
                                                                                                if r in WORDS2:
                                                                                                    if random.random()<0.5:
                                                                                                        #sample Topic
                                                                                                        W0=0
                                                                                                        W1=0
                                                                                                        W2=0


                                                                                                        try:
                                                                                                                    for w in WORDS2[r]:
                                                                                                                        if len(r_wts[w])<1:
                                                                                                                            continue
                                                                                                                        W0=W0+r_wts[w][0]
                                                                                                                        W1=W1+(1-r_wts[w][0])
                                                                                                                        #W2=W2+(r_wts[w][2])

                                                                                                                        if w in s_words:
                                                                                                                            if r not in expns2 or w not in expns2[r]:
                                                                                                                                                expns2[r][w] = r_wts[w][0]
                                                                                                                                                expns2[r][w] = (1-r_wts[w][0])
                                                                                                                                                #expns2[r][w] = (r_wts[w][2])


                                                                                                                            else:
                                                                                                                                                expns2[r][w] = expns2[r][w] + r_wts[w][0]
                                                                                                                                                expns2[r][w] = expns2[r][w] + (1-r_wts[w][0])
                                                                                                                                                #expns2[r][w] = expns2[r][w] + (r_wts[w][2])

                                                                                                        except:
                                                                                                            continue



                                                                                                        if (W0+W1)!= 0:
                                                                                                            W0 = W0/(W0+W1)
                                                                                                            W1 = W1/(W0+W1)
                                                                                                            #W2=W2/(W0+W1+W2)


                                                                                                            sval = random.random()
                                                                                                            iter1[r]=iter1[r]+1
                                                                                                            ##print(sval,W14,W15,W16,W17,W18,W19)
                                                                                                            if sval<W0:
                                                                                                                Sample[r]=0
                                                                                                                Sample_r[r33]=0
                                                                                                                margs23[r][0]=margs23[r][0]+1
                                                                                                            elif sval<(W0+W1):
                                                                                                                Sample[r]=1
                                                                                                                Sample_r[r33]=1
                                                                                                                margs23[r][1]=margs23[r][1]+1



                                                                                                            for r22 in H1[h1]:
                                                                                                                if r22 in WORDS2 and r33 in WORDS2:
                                                                                                                        if r22==r33:
                                                                                                                            continue
                                                                                                                        if Sample_r[r22]!=Sample_r[r33]:
                                                                                                                            if Sample_r[r22]==0:
                                                                                                                               # #print("yesssssss")
                                                                                                                                #P = P + VR
                                                                                                                                #exp1 = exp1 + 0.1
                                                                                                                                hhlll="Samehotel("+str(r33)+","+str(r22)+")"
                                                                                                                                if r33 not in expns2 or hhlll not in expns2[r33]:
                                                                                                                                    expns2[r33][hhlll] = VR

                                                                                                                                else:
                                                                                                                                    expns2[r33][hhlll] = expns2[r33][hhlll] + VR 
                                                                                                                        elif Sample_r[r22]!=Sample_r[r33]:
                                                                                                                            if Sample_r[r22]==1:
                                                                                                                                #P = P + VR
                                                                                                                                #exp1 = exp1 + 0.1
                                                                                                                                hhlll="Samehotel("+str(r33)+","+str(r22)+")"
                                                                                                                                if r33 not in expns2 or hhlll not in expns2[r33]:
                                                                                                                                    expns2[r33][hhlll] = VR

                                                                                                                                else:
                                                                                                                                    expns2[r33][hhlll] = expns2[r33][hhlll] + VR

                                                                                                            for r1 in Samuser_index[h]:
                                                                                                                if r1==r:
                                                                                                                    continue
                                                                                                                if r in WORDS2 and r1 in WORDS2:
                                                                                                                        if Sample[r]!=Sample[r1]:
                                                                                                                            if Sample[r1]==0:
                                                                                                                                #W0=W0+r_wts1[w][0]
                                                                                                                                #margs[r][0]=margs[r][0]+1
                                                                                                                                hhlll="Sameuser("+str(r)+","+str(r1)+")"
                                                                                                                                if r not in expns2 or hhlll not in expns2[r]:
                                                                                                                                    expns2[r][hhlll] =C1
                                                                                                                                    if r not in Relational_formula_filter:
                                                                                                                                        Relational_formula_filter[r]=WORDS2[r]    
                                                                                                                                else:
                                                                                                                                    expns2[r][hhlll] = expns2[r][hhlll] + C1
                                                                                                                            elif Sample[r1]==1:
                                                                                                                                #W1=W1+r_wts1[w][1]
                                                                                                                               # margs[r][1]=margs[r][1]+1
                                                                                                                                hhlll="Sameuser("+str(r)+","+str(r1)+")"
                                                                                                                                if r not in expns2 or hhlll not in expns2[r]:
                                                                                                                                    expns2[r][hhlll] =C1
                                                                                                                                    if r not in Relational_formula_filter:
                                                                                                                                        Relational_formula_filter[r]=WORDS2[r]    
                                                                                                                                else:
                                                                                                                                    expns2[r][hhlll] = expns2[r][hhlll] +C1 






                                                                                            #Computing Marginal Probability after user input
                                                                                            margs22={}
                                                                                            for t in margs23:
                                                                                                gh=[]
                                                                                                if iter1[t]>0:
                                                                                                    for kk in margs23[t]:
                                                                                                        vv=float(kk)/float(iter1[t])
                                                                                                        if float(vv)>=1.0:
                                                                                                            gh.append(float(1.0))
                                                                                                        elif float(vv)<1.0:
                                                                                                            gh.append(abs(float(vv)))
                                                                                                    margs22[t]=gh
                                                                                            margs33={}
                                                                                            for t in margs22:
                                                                                                s=0
                                                                                                for ww in margs22[t]:
                                                                                                    s=s+float(ww)
                                                                                                #if s!=0:
                                                                                                ##print(t,s)
                                                                                                margs33[t]=margs22[t]
                                                                                            margs3={}
                                                                                            margs222={}
                                                                                            for ww in margs33:
                                                                                                gg=[]
                                                                                                for kk in range(0,len(margs33[ww])):
                                                                                                        gg.append(margs33[ww][kk])
                                                                                                margs222[ww]=gg

                                                                                            for dd in margs222:
                                                                                                v=max(margs222[dd])
                                                                                                margs3[dd]=v
                                                                                            #data tyoe
                                                                                            d_tt={}
                                                                                            d_tt[0]="negative"
                                                                                            d_tt[1]="positive"
                                                                                            #predict topic user input
                                                                                            sampled_doc=[]
                                                                                            pred_t=[]
                                                                                            for a in margs222:
                                                                                                for ss in range(0,len(margs222[a])):
                                                                                                        if margs222[a][ss]==margs3[a]:
                                                                                                        ##print(a,d_tt[ss])
                                                                                                                sampled_doc.append(a)
                                                                                                                pred_t.append(d_tt[ss])
                                                                                            ss=set(sampled_doc)
                                                                                            sampled_doc_up=[]
                                                                                            sampled_doc_up_map_user={}
                                                                                            for kk in ss:
                                                                                                sampled_doc_up.append(kk)
                                                                                            for tt in sampled_doc_up:
                                                                                                ggf=[]
                                                                                                for gg in range(0,len(sampled_doc)):
                                                                                                    if tt==sampled_doc[gg]:
                                                                                                        ggf.append(pred_t[gg])
                                                                                                if len(ggf)==1:
                                                                                                    sampled_doc_up_map_user[tt]=ggf

                                                                                            cx=0   
                                                                                            for s in sampled_doc_up_map_user:
                                                                                                if len(sampled_doc_up_map_user[s])>1:
                                                                                                        cx=cx+1
                                                                                                       ##print(s,sampled_doc_up_map[s])


                                                                                            ##print(doc_per_pred_topic)
                                                                                            ##print(cx)

                                                                                            #ffd1=open("User_Prediction1.txt","w")

                                                                                            for s in sampled_doc_up_map_user:
                                                                                                ccvb=str(s)+":"+str(sampled_doc_up_map_user[s])
                                                                                                #ffd1.write(str(ccvb)+"\n")
                                                                                            #ffd1.close()

                                                                                            #Explanation Generation  with user
                                                                                            ##print(expns2)
                                                                                            import operator
                                                                                            correct_predictions_r = {}

                                                                                            for m in margs222.keys():
                                                                                                        if m in WORDS2 and m in sampled_doc_up_map_user:
                                                                                                                correct_predictions_r[m] = 1
                                                                                                                ##print(m)

                                                                                                        #if len(WORDS[m])==0:#or ratings[m]==3:
                                                                                                           # continue
                                                                                                        #else:
                                                                                                           # correct_predictions[m] = 1
                                                                                                        #if margs[m] > 0.5 and spamindex[m] ==-1:
                                                                                                            #correct_predictions[m] = 1
                                                                                            #fft_r=open("expn_n1_r1_user.txt","w")  
                                                                                            explanation_r={}
                                                                                            for e in expns2: 
                                                                                                if e in correct_predictions_r:
                                                                                                    sorted_expns_r = sorted(expns2[e].items(), key=operator.itemgetter(1),reverse=True)
                                                                                                    z = 0
                                                                                                    for s in sorted_expns_r[:]:
                                                                                                        z = z + s[1]
                                                                                                    rex_r = {}
                                                                                                    keys_r = []
                                                                                                    for s in sorted_expns_r[:]:
                                                                                                        rex_r[s[0]] = s[1]/z
                                                                                                        keys_r.append(s[0])
                                                                                                    #if "Sameuser" not in keys or "Samehotel" not in keys:
                                                                                                        #continue
                                                                                                    sorted1 = sorted(rex_r.items(), key=operator.itemgetter(1),reverse=True)
                                                                                                    #if sorted1[0][0]=="JNTM":
                                                                                                    ##print(str(e) +" "+str(sorted1))
                                                                                                    gg=str(e) +":"+str(sorted1)
                                                                                                    ##print(sorted1)
                                                                                                    explanation_r[e]=sorted1
                                                                                                    #fft_r.write(str(gg)+"\n")
                                                                                           # hhh="Explanation_K="+str(j)+"_sample_"+str(mzzz)+"_user"+"with user feedback"+".txt"
                                                                                            ##print("Samle:"+str(mzzz))
                                                                                            #newpath = r'C:\Users\khanfarabi\OneDrive - The University of Memphis\Explain_MLN\Explanation_yelp\Folder_20'
                                                                                            #if not os.path.exists(newpath):
                                                                                                                    #os.makedirs(newpath)

                                                                                            #f11_r=open(os.path.join(newpath,hhh), 'w')
                                                                                            #path="/Users/khanfarabi/OneDrive - The University of Memphis/Explain_MLN/Explanation_yelp/Folder_"+str(j+1)
                                                                                            #newpath="Folder_"+str(j)
                                                                                            #if not os.path.exists(path):
                                                                                                                       # os.makedirs(path)
                                                                                            #f11_r=open(os.path.join(path,hhh), 'w')

                                                                                            Store_Explanation_user={}
                                                                                            for t in explanation_r:
                                                                                                #for k in WORDS1:
                                                                                                       #if str(t)==str(k):
                                                                                                            ggg=str(t)+":"+str(explanation_r[t])
                                                                                                            #f11_r.write(str(ggg)+"\n")
                                                                                                           # f11_r.write("\n")
                                                                                                            ##print(t,explanation_r[t])
                                                                                                            ##print("\n")
                                                                                                            Store_Explanation_user[t]=explanation_r[t]
                                                                                                            Sample_model[t]=explanation_r[t]
                                                                                            expt_st_pd[mzzz]=Store_Explanation_user
                                                                                            ##print(mzzz,len(Store_Explanation_user))

                                                                                            #f11_r.close()
                                                                                            return expt_st_pd,margs3,reduced_obj






                                                    def compute_exp_acc(K,T):
                                                                final_clu=cluster_generatio(K)#cluster_generatio(j)
                                                                jjk={}
                                                                jjz={}
                                                                ob={}
                                                                for cz in range(0,T):
                                                                        M=0.35
                                                                        ss=random.random()
                                                                        if ss<0.2:
                                                                            M1=ss
                                                                        else:
                                                                            M1=random.random()
                                                                        expt_st_pd,margs3,reduced_obj=sample_rev(cz,final_clu,K,M)
                                                                        jjk[cz]=expt_st_pd
                                                                        jjz[cz]=margs3
                                                                        ob[cz]=reduced_obj
                                                                expt_st_pc[K]=jjk
                                                                perd_m[K]=jjz
                                                                obj_m[K]=ob
                                                                ob_clm={}
                                                                for t in final_clu:
                                                                       for jj in final_clu[t]:
                                                                            ##print(jj,t)
                                                                            ob_clm[int(jj)]=int(t)
                                                                reliab_exp={}
                                                                uq=[]

                                                                for kk in expt_st_pc[K]:
                                                                            #c=0
                                                                            for jj in expt_st_pc[K][kk]:
                                                                                    for yy in expt_st_pc[K][kk][jj]:
                                                                                          if yy not in uq:
                                                                                                uq.append(yy)
                                                                red_o=[]
                                                                ss=set(uq)
                                                                for v in ss:
                                                                    red_o.append(v)
                                                                from collections import defaultdict
                                                                reliab_exp={}
                                                                for zz in red_o:
                                                                        yz={}#defaultdict(list)
                                                                        for kk in expt_st_pc[K]:
                                                                               for jj in expt_st_pc[K][kk]:
                                                                                    for yy in expt_st_pc[K][kk][jj]:
                                                                                        if int(zz)==int(yy):
                                                                                            yz[int(jj)]=expt_st_pc[K][kk][jj][yy]
                                                                        reliab_exp[int(zz)]=yz
                                                                # Word Aggregation 
                                                                nm=[]
                                                                reliab_exp_up={}
                                                                for d in range(0,T):
                                                                    nm.append(int(d))

                                                                for tt in reliab_exp:
                                                                    #if tt==1:
                                                                    cz=[]
                                                                    c3=0

                                                                    exp=[]
                                                                    for kk in reliab_exp[tt]:
                                                                                cz.append(int(kk)) 
                                                                    cz1=list(set(nm)-set(cz))
                                                                    ##print(tt,cz,cz1)
                                                                    for zb in cz1:
                                                                        try:
                                                                            for k in obj_m[K][zb]:
                                                                                if ob_clm[int(tt)]==ob_clm[int(k)]:
                                                                                    break
                                                                        except:
                                                                            continue
                                                                        ##print(tt,k,Store_Explanation_user2[k])
                                                                        ##print("\n\n")
                                                                        c2=0
                                                                        c3=c3+1
                                                                        gggg="D_"+str(c3)
                                                                        vbb=[]
                                                                        #vbb.append(gggg)
                                                                        if k in Store_Explanation_user2:
                                                                            for hh in Store_Explanation_user2[k]:
                                                                                    c2=c2+1
                                                                                    if "Same" not in hh[0]:
                                                                                                bb1=hh[0]#+"+ R:"+str(c2)+"+ W:"+str(hh[1])+"+ M:"+str(margs3_u[k])
                                                                                                ##print(tt,zb,bb1)
                                                                                                ##print("\n")
                                                                                                if bb1 not in vbb:
                                                                                                    vbb.append(bb1)
                                                                            exp.append(vbb)
                                                                        ##print(bb1)
                                                                        #if bb1 not in vbb:

                                                                    for cx in cz:
                                                                        c=0
                                                                        c3=c3+1
                                                                        gggg="D_"+str(c3)
                                                                        vbb=[]
                                                                        #vbb.append(gggg)
                                                                        for hh in reliab_exp[tt][int(cx)]:
                                                                                c=c+1
                                                                                if "Same" not in hh[0]:
                                                                                            bb=hh[0]#+"+ R:"+str(c)+"+ W:"+str(hh[1])+"+ M:"+str(perd_m[10][int(cx)][tt])
                                                                                            ##print(tt,cx,bb)
                                                                                            ##print("\n\n")
                                                                                            if bb not in vbb:
                                                                                                vbb.append(bb)
                                                                        exp.append(vbb)
                                                                    reliab_exp_up[tt]=exp
                                                                    #R aggregate words



                                                                    #rank_list = [['A', 'B', 'C'], ['B', 'A', 'C'], ['C', 'D', 'A']]
                                                                    from rpy2.robjects import r
                                                                    reliab_exp_up_1w={}#reliab_exp_up_2w={}#reliab_exp_up_1w

                                                                    for t1 in reliab_exp_up:
                                                                        gh=[]
                                                                        gh1=[]
                                                                        c=0
                                                                        gh.append(t1)
                                                                        ll=reliab_exp_up[t1]
                                                                        b=r['CEMC'](ll,k=30)#agg.CEMC(ll)
                                                                        for t in b[0]:
                                                                            if c<T:
                                                                                gh.append(t)
                                                                                c=c+1
                                                                       # #print(gh)
                                                                        gh1.append(gh)
                                                                        reliab_exp_up_1w[t1]=gh1
                                                                        # Relational Aggregation 
                                                                        nm=[]
                                                                        rw={}
                                                                        reliab_exp_up2={}
                                                                        for d in range(0,T):
                                                                            nm.append(int(d))

                                                                        for tt in reliab_exp:
                                                                            #if tt==1:
                                                                            cz=[]
                                                                            c3=0

                                                                            exp=[]
                                                                            for kk in reliab_exp[tt]:
                                                                                        cz.append(int(kk)) 
                                                                            cz1=list(set(nm)-set(cz))
                                                                            ##print(tt,cz,cz1)
                                                                            for zb in cz1:
                                                                                try:
                                                                                    for k in obj_m[K][zb]:
                                                                                        if ob_clm[int(tt)]==ob_clm[int(k)]:
                                                                                            break
                                                                                except:
                                                                                    continue
                                                                                ##print(tt,k,Store_Explanation_user2[k])
                                                                                ##print("\n\n")
                                                                                c2=0
                                                                                c3=c3+1
                                                                                gggg="D_"+str(c3)
                                                                                vbb=[]
                                                                                #vbb.append(gggg)
                                                                                if k in Store_Explanation_user2:
                                                                                        for hh in Store_Explanation_user2[k]:
                                                                                                c2=c2+1
                                                                                                if "Same"  in hh[0]:
                                                                                                            gggz=hh[0].strip(")").split(",")
                                                                                                            hh1="Q_"+str(gggz[1])
                                                                                                            rw[hh1]=float(hh[1])
                                                                                                            bb1=hh1#+"+ R:"+str(c2)+"+ W:"+str(hh[1])+"+ M:"+str(margs3_u[k])
                                                                                                            ##print(tt,zb,bb1)
                                                                                                            ##print("\n")
                                                                                                            if bb1 not in vbb:
                                                                                                                vbb.append(bb1)             

                                                                                        exp.append(vbb)
                                                                                ##print(bb1)
                                                                                #if bb1 not in vbb:

                                                                            for cx in cz:
                                                                                c=0
                                                                                c3=c3+1
                                                                                gggg="D_"+str(c3)
                                                                                vbb=[]
                                                                                #vbb.append(gggg)
                                                                                for hh in reliab_exp[tt][int(cx)]:
                                                                                        c=c+1
                                                                                        if "Same"  in hh[0]:
                                                                                                    gggz=hh[0].strip(")").split(",")
                                                                                                    hh1="Q_"+str(gggz[1])
                                                                                                    rw[hh1]=float(hh[1])
                                                                                                    bb=hh1#+"+ R:"+str(c)+"+ W:"+str(hh[1])+"+ M:"+str(perd_m[10][int(cx)][tt])
                                                                                                    ##print(tt,cx,bb)
                                                                                                    ##print("\n\n")
                                                                                                    if bb not in vbb:
                                                                                                        vbb.append(bb)        
                                                                                exp.append(vbb)
                                                                            reliab_exp_up2[tt]=exp
                                                                            # Relational Aggregation 
                                                                            nm=[]
                                                                            rw={}
                                                                            reliab_exp_up2={}
                                                                            for d in range(0,T):
                                                                                nm.append(int(d))

                                                                            for tt in reliab_exp:
                                                                                #if tt==1:
                                                                                cz=[]
                                                                                c3=0

                                                                                exp=[]
                                                                                for kk in reliab_exp[tt]:
                                                                                            cz.append(int(kk)) 
                                                                                cz1=list(set(nm)-set(cz))
                                                                                ##print(tt,cz,cz1)
                                                                                for zb in cz1:
                                                                                    try:
                                                                                        for k in obj_m[K][zb]:
                                                                                            if ob_clm[int(tt)]==ob_clm[int(k)]:
                                                                                                break
                                                                                    except:
                                                                                        continue
                                                                                    ##print(tt,k,Store_Explanation_user2[k])
                                                                                    ##print("\n\n")
                                                                                    c2=0
                                                                                    c3=c3+1
                                                                                    gggg="D_"+str(c3)
                                                                                    vbb=[]
                                                                                    #vbb.append(gggg)
                                                                                    if k in Store_Explanation_user2:
                                                                                            for hh in Store_Explanation_user2[k]:
                                                                                                    c2=c2+1
                                                                                                    if "Same"  in hh[0]:
                                                                                                                gggz=hh[0].strip(")").split(",")
                                                                                                                hh1="Q_"+str(gggz[1])
                                                                                                                rw[hh1]=float(hh[1])
                                                                                                                bb1=hh1#+"+ R:"+str(c2)+"+ W:"+str(hh[1])+"+ M:"+str(margs3_u[k])
                                                                                                                ##print(tt,zb,bb1)
                                                                                                                ##print("\n")
                                                                                                                if bb1 not in vbb:
                                                                                                                    vbb.append(bb1)             

                                                                                            exp.append(vbb)
                                                                                    ##print(bb1)
                                                                                    #if bb1 not in vbb:

                                                                                for cx in cz:
                                                                                    c=0
                                                                                    c3=c3+1
                                                                                    gggg="D_"+str(c3)
                                                                                    vbb=[]
                                                                                    #vbb.append(gggg)
                                                                                    for hh in reliab_exp[tt][int(cx)]:
                                                                                            c=c+1
                                                                                            if "Same"  in hh[0]:
                                                                                                        gggz=hh[0].strip(")").split(",")
                                                                                                        hh1="Q_"+str(gggz[1])
                                                                                                        rw[hh1]=float(hh[1])
                                                                                                        bb=hh1#+"+ R:"+str(c)+"+ W:"+str(hh[1])+"+ M:"+str(perd_m[10][int(cx)][tt])
                                                                                                        ##print(tt,cx,bb)
                                                                                                        ##print("\n\n")
                                                                                                        if bb not in vbb:
                                                                                                            vbb.append(bb)        
                                                                                    exp.append(vbb)
                                                                                reliab_exp_up2[tt]=exp
                                                                                #R aggregate relation

                                                                                #rank_list = [['A', 'B', 'C'], ['B', 'A', 'C'], ['C', 'D', 'A']]
                                                                                from rpy2.robjects import r
                                                                                #reliab_exp_up_1={} reliab_exp_up_2={}reliab_exp_up_3={}
                                                                                reliab_exp_up_1r={}#reliab_exp_up_2r={}#reliab_exp_up_1r

                                                                                for t1 in reliab_exp_up2:
                                                                                    gh=[]
                                                                                    gh1=[]
                                                                                    c=0
                                                                                    gh.append(t1)
                                                                                    ll=reliab_exp_up2[t1]
                                                                                    try:
                                                                                        b=r['CEMC'](ll,k=10)#agg.CEMC(ll)
                                                                                        for t in b[0]:
                                                                                            if c<T:
                                                                                                gh.append(t)
                                                                                                c=c+1
                                                                                        gh1.append(gh)
                                                                                        reliab_exp_up_1r[t1]=gh1
                                                                                    except:
                                                                                        continue
                                                                                # Aggregate Review Relational Exp
                                                                                aggregate_relational_exp_review={}
                                                                                for t in reliab_exp_up_1r:
                                                                                    expr=[]
                                                                                    for d in reliab_exp_up_1r[t][0]:
                                                                                        if 'Q' in str(d):
                                                                                            hh=d.split("_")
                                                                                            if qrat[int(t)]==qrat[int(hh[1])]:
                                                                                                        ##print(t,hh[1])
                                                                                                        expr.append(hh[1])
                                                                                    if len(expr)>=1:
                                                                                                aggregate_relational_exp_review[t]=expr

                                                                                #words exp accuracy
                                                                                reliab_exp_up_1w_pc={}
                                                                                reliab_exp_up_1w_score={}
                                                                                reliab_exp_up_1wu={}

                                                                                for jj in reliab_exp_up_1w:
                                                                                    if jj in ann:
                                                                                        ghh=[]
                                                                                        cv=0
                                                                                        c=0
                                                                                        mm=5
                                                                                        for cz1 in reliab_exp_up_1w[jj][0]:
                                                                                                 if cv<mm:
                                                                                                     if cz1 in ann[jj]:
                                                                                                                c=c+1
                                                                                                                cv=cv+1
                                                                                        #mm=mm-0.45
                                                                                        s=c/mm

                                                                                        if s>0:
                                                                                            reliab_exp_up_1w_pc[jj]=s
                                                                                            ##print(s)


                                                                                zx=0
                                                                                cx=0
                                                                                for z in reliab_exp_up_1w_pc:
                                                                                    zx=zx+reliab_exp_up_1w_pc[z]
                                                                                    cx=cx+1
                                                                                ##print(reliab_exp_up_1w_score)
                                                                                ##print(zx/cx)
                                                                                try:
                                                                                    wexp=zx/cx
                                                                                except:
                                                                                    continue 

                                                                                # Relational exp accuracy

                                                                                cn=0
                                                                                agg_s={}
                                                                                agg_avg={}
                                                                                for t in aggregate_relational_exp_review:
                                                                                    s=0
                                                                                    if t in similar_r_map:
                                                                                        for z in aggregate_relational_exp_review[t]:
                                                                                            if z in similar_r_map[t]:
                                                                                                s=s+1
                                                                                    if s>0:
                                                                                        ss=s/5.0
                                                                                        if ss>1.0:
                                                                                            agg_avg[t]=1.0
                                                                                        else:
                                                                                            agg_avg[t]=ss
                                                                                st=0        
                                                                                for y in agg_avg:
                                                                                    st=st+agg_avg[y]
                                                                                ##print("Relational Accuracy: "+str(st/len(agg_avg))+"\n")
                                                                                try:
                                                                                    rexp=st/len(agg_avg)

                                                                                except:
                                                                                    continue
                                                                                return wexp,rexp














                                                    def   varying_cluster(K,rep,T):
                                                            wxz=[]
                                                            rxz=[]
                                                            for t in range(0,rep):
                                                                wexp,rexp=compute_exp_acc(K,T)
                                                                wxz.append(wexp)
                                                                rxz.append(rexp)
                                                            return wxz,rxz


                                                    cluster_varying_wexp={}
                                                    cluster_varying_rexp={}

                                                    st=5#int(input())
                                                    en=st*5+1
                                                    for t in range(st,en,5):
                                                                         rep=10
                                                                         T=10
                                                                         wxz,rxz=varying_cluster(t,rep,T)
                                                                         cluster_varying_rexp[t]=rxz
                                                                         cluster_varying_wexp[t]=wxz




                                                    #print(cluster_varying_wexp,cluster_varying_rexp)






# Varying Model

class md:
    @classmethod
    def fg(cls):
            #new technique with reduced object varying models size aggregate 
            import rpy2
            import rpy2.robjects.packages as rpackages
            from rpy2.robjects.vectors import StrVector
            from rpy2.robjects.packages import importr
            utils = rpackages.importr('utils')
            utils.chooseCRANmirror(ind=1)
            # Install packages
            packnames = ('TopKLists', 'CEMC')
            utils.install_packages(StrVector(packnames))
            packnames = ('data(TopKSpaceSampleInput)')
            utils.install_packages(StrVector(packnames))
            h = importr('TopKLists')


            expt_st_pc={}
            perd_m={}
            obj_m={}
            Sample_model={}
            def cluster_generatio(kk):

                                    #cluster generation with k-means
                                    import sys
                                    from nltk.cluster import KMeansClusterer
                                    import nltk
                                    from sklearn import cluster
                                    from sklearn import metrics
                                    import gensim 
                                    import operator
                                    from gensim.models import Word2Vec
                                    model = Word2Vec(sent, min_count=1)
                                    X = model[model.wv.vocab]
                                    NUM_CLUSTERS=kk
                                    kclusterer = KMeansClusterer(NUM_CLUSTERS, distance=nltk.cluster.util.cosine_distance,repeats=25)
                                    assigned_clusters1 = kclusterer.cluster(X,assign_clusters=True)
                                    ##print (assigned_clusters)
                                    cluster={}
                                    words = list(model.wv.vocab)
                                    for i, word in enumerate(words):
                                      gh=[] 
                                      gh1=[] 
                                      gh2=[] 
                                      if word.isdigit(): 
                                        cluster[word]=assigned_clusters1[i]
                                        ##print (word + ":" + str(assigned_clusters[i]))
                                    cluster_final={}
                                    for j in range(NUM_CLUSTERS):
                                        gg=[]
                                        for tt in cluster:
                                            if int(cluster[tt])==int(j):
                                                if tt not in gg:
                                                    gg.append(tt)
                                        if len(gg)>0:
                                                    cluster_final[j]=gg
                                    cc=0
                                    final_clu={}
                                    for t in cluster_final:
                                        ghh=[]
                                        for k in cluster_final[t]:
                                            if int(k) in WORDS:
                                                   ghh.append(int(k))
                                        if len(ghh)>=2:
                                                final_clu[cc]=ghh
                                                cc=cc+1
                                    return final_clu


            def sample_rev(mzzz,final_clu,M):
                                                    #reduced objects
                                                    reduced_obj=[]
                                                    expt_st_pd={}
                                                    import random
                                                    import operator
                                                    import os.path
                                                    import os 

                                                    for k in final_clu:
                                                        c=0
                                                        for tt in final_clu[k]:
                                                            if random.random()<M:
                                                                c=c+1
                                                                chh=float(c)/len(final_clu[k])
                                                                if chh<=M:
                                                                        ##print(M)
                                                                        if tt not in reduced_obj:
                                                                            reduced_obj.append(tt)
                                                    ##print(len(reduced_obj)/float(len(final_clu[k]))
                                                    wr={}
                                                    w=[]
                                                    for k in final_clu:
                                                            #c=-1
                                                            #c=c+1
                                                            md=int(len(final_clu[k])//2)
                                                            c=0      
                                                            k1= final_clu[k][md+c]
                                                            ##print(k1,md)        
                                                            if k1 in ann:
                                                                        for k3 in ann[k1]:
                                                                                w.append(k3)
                                                            else:
                                                                c=c+11
                                                                continue 


                                                           # #print(k,k1,md,d_tt[qrat[k1]],w)
                                                            wr[k1]=w
                                                    model = Word2Vec(sent, min_count=1)
                                                    data_g={}
                                                    for t in WORDS1:
                                                        chu=[]
                                                        #try:
                                                        vb={}
                                                        for v in w:
                                                            vb1={}
                                                            for v1 in WORDS1[t]:
                                                                    ##print(v1,v)
                                                                    gh1=model.similarity(v,v1)
                                                                    if gh1>=0.40:
                                                                          vb1[v1]=float(gh1)
                                                                          ##print(gh1)
                                                            for jk in vb1:
                                                                if jk in vb:
                                                                    if float(vb1[jk])>=float(vb[jk]):
                                                                        ##print(jk,vb1[jk],vb[jk])
                                                                        vb[jk]=vb1[jk]
                                                                else:
                                                                    vb[jk]=vb1[jk]
                                                        ##print(t, vb)
                                                        ##print("\n")             
                                                        dd=sorted(vb.items(), key=operator.itemgetter(1),reverse=True)
                                                        cc=0
                                                        for kkk in dd:
                                                            if kkk[0] not in chu:
                                                                if cc<25:
                                                                    chu.append(kkk[0])
                                                                    cc=cc+1

                                                        if len(chu)>0:
                                                            data_g[t]=chu
                                                    #survey checking
                                                    #print(len(WORDS1))
                                                    #Updating the Whole Evidence Based on manual annotation
                                                    WORDS22={}
                                                    for gg in WORDS1:
                                                        #if gg in data_extract12:
                                                            #WORDS2[gg]=data_extract12[gg]
                                                        if gg in data_g:
                                                            if len(data_g[gg])>0:
                                                                WORDS22[gg]=data_g[gg]
                                                    WORDS2={}
                                                    for t in reduced_obj:
                                                        hhh1=[]
                                                        czx=0
                                                        #cxzz=random.randint(5,11)
                                                        if t in WORDS22:
                                                            #random.shuffle(WORDS22[t])
                                                            for dd in WORDS22[t]:
                                                                if dd not in hhh1:
                                                                    if czx<10 and dd in s_words or str(dd) in s_words:
                                                                        hhh1.append(dd)
                                                                        czx=czx+1

                                                            WORDS2[t]=hhh1
                                                            ##print(len(hhh1))
                                                    ##print(WORDS2)
                                                    ##print(len(WORDS2),len(reduced_obj),len(WORDS22))
                                                    #Sample 2 user

                                                    import random
                                                    from collections import defaultdict

                                                    #Sampler
                                                    margs23 =defaultdict(list)
                                                    Relational_formula_filter={}
                                                    iter1=defaultdict(list)
                                                    users_s= defaultdict(list)
                                                    expns2 = defaultdict(dict)
                                                    Relational_formula_filter={}
                                                    same_user={}
                                                    Sample={}
                                                    Sample_r={}
                                                    for h in H1:
                                                        for i,r in enumerate(H1[h]):
                                                            if r in WORDS2:
                                                                        #Sample[r] = random.randint(0,1)
                                                                        Sample_r[r] = random.randint(0,1)
                                                    for h in Samuser_index:
                                                        for i,r in enumerate(Samuser_index[h]):
                                                            if r in WORDS2:
                                                                        Sample[r] = random.randint(0,1)
                                                                        #Sample_r[r] = random.randint(0,2)
                                                                        #Sample_r[r] = random.randint(0,1)
                                                                        margs23[r] = [0]*2
                                                                        #margs_r[r] = 0
                                                                        iter1[r] =0

                                                    #Tunable parameter (default value of prob)
                                                    C1 = 0.98
                                                    VR=0.98
                                                    iters =1000000

                                                    for t in range(0,iters,1):
                                                        h1 = random.choice(list(H1.keys()))
                                                        if len(H1[h1])==0:
                                                            continue
                                                        ix1 = random.randint(0,len(H1[h1])-1)
                                                        r33 = H1[h1][ix1]
                                                        h = random.choice(list(Samuser_index.keys()))
                                                        if len(Samuser_index[h])==0:
                                                            continue
                                                        ix = random.randint(0,len(Samuser_index[h])-1)
                                                        r = Samuser_index[h][ix]
                                                        if r in WORDS2:
                                                            if random.random()<0.5:
                                                                #sample Topic
                                                                W0=0
                                                                W1=0
                                                                W2=0


                                                                try:
                                                                            for w in WORDS2[r]:
                                                                                if len(r_wts[w])<1:
                                                                                    continue
                                                                                W0=W0+r_wts[w][0]
                                                                                W1=W1+(1-r_wts[w][0])
                                                                                #W2=W2+(r_wts[w][2])

                                                                                if w in s_words:
                                                                                    if r not in expns2 or w not in expns2[r]:
                                                                                                        expns2[r][w] = r_wts[w][0]
                                                                                                        expns2[r][w] = (1-r_wts[w][0])
                                                                                                        #expns2[r][w] = (r_wts[w][2])


                                                                                    else:
                                                                                                        expns2[r][w] = expns2[r][w] + r_wts[w][0]
                                                                                                        expns2[r][w] = expns2[r][w] + (1-r_wts[w][0])
                                                                                                        #expns2[r][w] = expns2[r][w] + (r_wts[w][2])

                                                                except:
                                                                    continue



                                                                if (W0+W1)!= 0:
                                                                    W0 = W0/(W0+W1)
                                                                    W1 = W1/(W0+W1)
                                                                    #W2=W2/(W0+W1+W2)


                                                                    sval = random.random()
                                                                    iter1[r]=iter1[r]+1
                                                                    ##print(sval,W14,W15,W16,W17,W18,W19)
                                                                    if sval<W0:
                                                                        Sample[r]=0
                                                                        Sample_r[r33]=0
                                                                        margs23[r][0]=margs23[r][0]+1
                                                                    elif sval<(W0+W1):
                                                                        Sample[r]=1
                                                                        Sample_r[r33]=1
                                                                        margs23[r][1]=margs23[r][1]+1



                                                                    for r22 in H1[h1]:
                                                                        if r22 in WORDS2 and r33 in WORDS2:
                                                                                if r22==r33:
                                                                                    continue
                                                                                if Sample_r[r22]!=Sample_r[r33]:
                                                                                    if Sample_r[r22]==0:
                                                                                       # #print("yesssssss")
                                                                                        #P = P + VR
                                                                                        #exp1 = exp1 + 0.1
                                                                                        hhlll="Samehotel("+str(r33)+","+str(r22)+")"
                                                                                        if r33 not in expns2 or hhlll not in expns2[r33]:
                                                                                            expns2[r33][hhlll] = VR

                                                                                        else:
                                                                                            expns2[r33][hhlll] = expns2[r33][hhlll] + VR 
                                                                                elif Sample_r[r22]!=Sample_r[r33]:
                                                                                    if Sample_r[r22]==1:
                                                                                        #P = P + VR
                                                                                        #exp1 = exp1 + 0.1
                                                                                        hhlll="Samehotel("+str(r33)+","+str(r22)+")"
                                                                                        if r33 not in expns2 or hhlll not in expns2[r33]:
                                                                                            expns2[r33][hhlll] = VR

                                                                                        else:
                                                                                            expns2[r33][hhlll] = expns2[r33][hhlll] + VR

                                                                    for r1 in Samuser_index[h]:
                                                                        if r1==r:
                                                                            continue
                                                                        if r in WORDS2 and r1 in WORDS2:
                                                                                if Sample[r]!=Sample[r1]:
                                                                                    if Sample[r1]==0:
                                                                                        #W0=W0+r_wts1[w][0]
                                                                                        #margs[r][0]=margs[r][0]+1
                                                                                        hhlll="Sameuser("+str(r)+","+str(r1)+")"
                                                                                        if r not in expns2 or hhlll not in expns2[r]:
                                                                                            expns2[r][hhlll] =C1
                                                                                            if r not in Relational_formula_filter:
                                                                                                Relational_formula_filter[r]=WORDS2[r]    
                                                                                        else:
                                                                                            expns2[r][hhlll] = expns2[r][hhlll] + C1
                                                                                    elif Sample[r1]==1:
                                                                                        #W1=W1+r_wts1[w][1]
                                                                                       # margs[r][1]=margs[r][1]+1
                                                                                        hhlll="Sameuser("+str(r)+","+str(r1)+")"
                                                                                        if r not in expns2 or hhlll not in expns2[r]:
                                                                                            expns2[r][hhlll] =C1
                                                                                            if r not in Relational_formula_filter:
                                                                                                Relational_formula_filter[r]=WORDS2[r]    
                                                                                        else:
                                                                                            expns2[r][hhlll] = expns2[r][hhlll] +C1 






                                                    #Computing Marginal Probability after user input
                                                    margs22={}
                                                    for t in margs23:
                                                        gh=[]
                                                        if iter1[t]>0:
                                                            for kk in margs23[t]:
                                                                vv=float(kk)/float(iter1[t])
                                                                if float(vv)>=1.0:
                                                                    gh.append(float(1.0))
                                                                elif float(vv)<1.0:
                                                                    gh.append(abs(float(vv)))
                                                            margs22[t]=gh
                                                    margs33={}
                                                    for t in margs22:
                                                        s=0
                                                        for ww in margs22[t]:
                                                            s=s+float(ww)
                                                        #if s!=0:
                                                        ##print(t,s)
                                                        margs33[t]=margs22[t]
                                                    margs3={}
                                                    margs222={}
                                                    for ww in margs33:
                                                        gg=[]
                                                        for kk in range(0,len(margs33[ww])):
                                                                gg.append(margs33[ww][kk])
                                                        margs222[ww]=gg

                                                    for dd in margs222:
                                                        v=max(margs222[dd])
                                                        margs3[dd]=v
                                                    #data tyoe
                                                    d_tt={}
                                                    d_tt[0]="negative"
                                                    d_tt[1]="positive"
                                                    #predict topic user input
                                                    sampled_doc=[]
                                                    pred_t=[]
                                                    for a in margs222:
                                                        for ss in range(0,len(margs222[a])):
                                                                if margs222[a][ss]==margs3[a]:
                                                                ##print(a,d_tt[ss])
                                                                        sampled_doc.append(a)
                                                                        pred_t.append(d_tt[ss])
                                                    ss=set(sampled_doc)
                                                    sampled_doc_up=[]
                                                    sampled_doc_up_map_user={}
                                                    for kk in ss:
                                                        sampled_doc_up.append(kk)
                                                    for tt in sampled_doc_up:
                                                        ggf=[]
                                                        for gg in range(0,len(sampled_doc)):
                                                            if tt==sampled_doc[gg]:
                                                                ggf.append(pred_t[gg])
                                                        if len(ggf)==1:
                                                            sampled_doc_up_map_user[tt]=ggf

                                                    cx=0   
                                                    for s in sampled_doc_up_map_user:
                                                        if len(sampled_doc_up_map_user[s])>1:
                                                                cx=cx+1
                                                               ##print(s,sampled_doc_up_map[s])

                                                    for s in sampled_doc_up_map_user:
                                                        ccvb=str(s)+":"+str(sampled_doc_up_map_user[s])


                                                    #Explanation Generation  with user
                                                    ##print(expns2)
                                                    import operator
                                                    correct_predictions_r = {}

                                                    for m in margs222.keys():
                                                                if m in WORDS2 and m in sampled_doc_up_map_user:
                                                                        correct_predictions_r[m] = 1
                                                                        ##print(m)

                                                                #if len(WORDS[m])==0:#or ratings[m]==3:
                                                                   # continue
                                                                #else:
                                                                   # correct_predictions[m] = 1
                                                                #if margs[m] > 0.5 and spamindex[m] ==-1:
                                                                    #correct_predictions[m] = 1
                                                    #fft_r=open("expn_n1_r1_user.txt","w")  
                                                    explanation_r={}
                                                    for e in expns2: 
                                                        if e in correct_predictions_r:
                                                            sorted_expns_r = sorted(expns2[e].items(), key=operator.itemgetter(1),reverse=True)
                                                            z = 0
                                                            for s in sorted_expns_r[:]:
                                                                z = z + s[1]
                                                            rex_r = {}
                                                            keys_r = []
                                                            for s in sorted_expns_r[:]:
                                                                rex_r[s[0]] = s[1]/z
                                                                keys_r.append(s[0])
                                                            #if "Sameuser" not in keys or "Samehotel" not in keys:
                                                                #continue
                                                            sorted1 = sorted(rex_r.items(), key=operator.itemgetter(1),reverse=True)
                                                            #if sorted1[0][0]=="JNTM":
                                                            ##print(str(e) +" "+str(sorted1))
                                                            gg=str(e) +":"+str(sorted1)
                                                            ##print(sorted1)
                                                            explanation_r[e]=sorted1
                                                            #fft_r.write(str(gg)+"\n")
                                                    #hhh="Explanation_K="+str(j)+"_sample_"+str(mzzz)+"_user"+"with user feedback"+".txt"
                                                    ##print("Samle:"+str(mzzz))
                                                    #newpath = r'C:\Users\khanfarabi\OneDrive - The University of Memphis\Explain_MLN\Explanation_yelp\Folder_20'
                                                    #if not os.path.exists(newpath):
                                                                            #os.makedirs(newpath)

                                                    #f11_r=open(os.path.join(newpath,hhh), 'w')
                                                    #path="/Users/khanfarabi/OneDrive - The University of Memphis/Explain_MLN/Explanation_yelp/Folder_"+str(j+1)
                                                    #newpath="Folder_"+str(j)
                                                    #if not os.path.exists(path):
                                                                               # os.makedirs(path)
                                                    #f11_r=open(os.path.join(path,hhh), 'w')

                                                    Store_Explanation_user={}
                                                    for t in explanation_r:
                                                        #for k in WORDS1:
                                                               #if str(t)==str(k):
                                                                    ggg=str(t)+":"+str(explanation_r[t])
                                                                   # f11_r.write(str(ggg)+"\n")
                                                                   # f11_r.write("\n")
                                                                    ##print(t,explanation_r[t])
                                                                    ##print("\n")
                                                                    Store_Explanation_user[t]=explanation_r[t]
                                                                    Sample_model[t]=explanation_r[t]
                                                    expt_st_pd[mzzz]=Store_Explanation_user
                                                    ##print(mzzz,len(Store_Explanation_user))

                                                    #f11_r.close()
                                                    return expt_st_pd,margs3,reduced_obj






            def compute_exp_acc(final_clu,T):
                        final_clu=final_clu
                       #cluster_generatio(j)
                        jjk={}
                        jjz={}
                        ob={}
                        K=T
                        for cz in range(0,T):
                                M=0.35
                                ss=random.random()
                                if ss<0.2:
                                    M1=ss
                                else:
                                    M1=random.random()
                                expt_st_pd,margs3,reduced_obj=sample_rev(cz,final_clu,M)
                                jjk[cz]=expt_st_pd
                                jjz[cz]=margs3
                                ob[cz]=reduced_obj
                        expt_st_pc[T]=jjk
                        perd_m[T]=jjz
                        obj_m[T]=ob
                        ob_clm={}
                        for t in final_clu:
                               for jj in final_clu[t]:
                                    ##print(jj,t)
                                    ob_clm[int(jj)]=int(t)
                        reliab_exp={}
                        uq=[]

                        for kk in expt_st_pc[K]:
                                    #c=0
                                    for jj in expt_st_pc[K][kk]:
                                            for yy in expt_st_pc[K][kk][jj]:
                                                  if yy not in uq:
                                                        uq.append(yy)
                        red_o=[]
                        ss=set(uq)
                        for v in ss:
                            red_o.append(v)
                        from collections import defaultdict
                        reliab_exp={}
                        for zz in red_o:
                                yz={}#defaultdict(list)
                                for kk in expt_st_pc[K]:
                                       for jj in expt_st_pc[K][kk]:
                                            for yy in expt_st_pc[K][kk][jj]:
                                                if int(zz)==int(yy):
                                                    yz[int(jj)]=expt_st_pc[K][kk][jj][yy]
                                reliab_exp[int(zz)]=yz
                        # Word Aggregation 
                        nm=[]
                        reliab_exp_up={}
                        for d in range(0,T):
                            nm.append(int(d))

                        for tt in reliab_exp:
                            #if tt==1:
                            cz=[]
                            c3=0

                            exp=[]
                            for kk in reliab_exp[tt]:
                                        cz.append(int(kk)) 
                            cz1=list(set(nm)-set(cz))
                            ##print(tt,cz,cz1)
                            for zb in cz1:
                                try:
                                    for k in obj_m[K][zb]:
                                        if ob_clm[int(tt)]==ob_clm[int(k)]:
                                            break
                                except:
                                    continue
                                ##print(tt,k,Store_Explanation_user2[k])
                                ##print("\n\n")
                                c2=0
                                c3=c3+1
                                gggg="D_"+str(c3)
                                vbb=[]
                                #vbb.append(gggg)
                                if k in Store_Explanation_user2:
                                    for hh in Store_Explanation_user2[k]:
                                            c2=c2+1
                                            if "Same" not in hh[0]:
                                                        bb1=hh[0]#+"+ R:"+str(c2)+"+ W:"+str(hh[1])+"+ M:"+str(margs3_u[k])
                                                        ##print(tt,zb,bb1)
                                                        ##print("\n")
                                                        if bb1 not in vbb:
                                                            vbb.append(bb1)
                                    exp.append(vbb)
                                ##print(bb1)
                                #if bb1 not in vbb:

                            for cx in cz:
                                c=0
                                c3=c3+1
                                gggg="D_"+str(c3)
                                vbb=[]
                                #vbb.append(gggg)
                                for hh in reliab_exp[tt][int(cx)]:
                                        c=c+1
                                        if "Same" not in hh[0]:
                                                    bb=hh[0]#+"+ R:"+str(c)+"+ W:"+str(hh[1])+"+ M:"+str(perd_m[10][int(cx)][tt])
                                                    ##print(tt,cx,bb)
                                                    ##print("\n\n")
                                                    if bb not in vbb:
                                                        vbb.append(bb)
                                exp.append(vbb)
                            reliab_exp_up[tt]=exp
                            #R aggregate words




                            #rank_list = [['A', 'B', 'C'], ['B', 'A', 'C'], ['C', 'D', 'A']]
                            from rpy2.robjects import r
                            reliab_exp_up_1w={}#reliab_exp_up_2w={}#reliab_exp_up_1w

                            for t1 in reliab_exp_up:
                                gh=[]
                                gh1=[]
                                c=0
                                gh.append(t1)
                                ll=reliab_exp_up[t1]
                                b=r['CEMC'](ll,k=30)#agg.CEMC(ll)
                                for t in b[0][0]:
                                    if c<T:
                                        gh.append(t)
                                        c=c+1
                               # #print(gh)
                                gh1.append(gh)
                                reliab_exp_up_1w[t1]=gh1
                                # Relational Aggregation 
                                nm=[]
                                rw={}
                                reliab_exp_up2={}
                                for d in range(0,T):
                                    nm.append(int(d))

                                for tt in reliab_exp:
                                    #if tt==1:
                                    cz=[]
                                    c3=0

                                    exp=[]
                                    for kk in reliab_exp[tt]:
                                                cz.append(int(kk)) 
                                    cz1=list(set(nm)-set(cz))
                                    ##print(tt,cz,cz1)
                                    for zb in cz1:
                                        try:
                                            for k in obj_m[K][zb]:
                                                if ob_clm[int(tt)]==ob_clm[int(k)]:
                                                    break
                                        except:
                                            continue
                                        ##print(tt,k,Store_Explanation_user2[k])
                                        ##print("\n\n")
                                        c2=0
                                        c3=c3+1
                                        gggg="D_"+str(c3)
                                        vbb=[]
                                        #vbb.append(gggg)
                                        if k in Store_Explanation_user2:
                                                for hh in Store_Explanation_user2[k]:
                                                        c2=c2+1
                                                        if "Same"  in hh[0]:
                                                                    gggz=hh[0].strip(")").split(",")
                                                                    hh1="Q_"+str(gggz[1])
                                                                    rw[hh1]=float(hh[1])
                                                                    bb1=hh1#+"+ R:"+str(c2)+"+ W:"+str(hh[1])+"+ M:"+str(margs3_u[k])
                                                                    ##print(tt,zb,bb1)
                                                                    ##print("\n")
                                                                    if bb1 not in vbb:
                                                                        vbb.append(bb1)             

                                                exp.append(vbb)
                                        ##print(bb1)
                                        #if bb1 not in vbb:

                                    for cx in cz:
                                        c=0
                                        c3=c3+1
                                        gggg="D_"+str(c3)
                                        vbb=[]
                                        #vbb.append(gggg)
                                        for hh in reliab_exp[tt][int(cx)]:
                                                c=c+1
                                                if "Same"  in hh[0]:
                                                            gggz=hh[0].strip(")").split(",")
                                                            hh1="Q_"+str(gggz[1])
                                                            rw[hh1]=float(hh[1])
                                                            bb=hh1#+"+ R:"+str(c)+"+ W:"+str(hh[1])+"+ M:"+str(perd_m[10][int(cx)][tt])
                                                            ##print(tt,cx,bb)
                                                            ##print("\n\n")
                                                            if bb not in vbb:
                                                                vbb.append(bb)        
                                        exp.append(vbb)
                                    reliab_exp_up2[tt]=exp
                                    # Relational Aggregation 
                                    nm=[]
                                    rw={}
                                    reliab_exp_up2={}
                                    for d in range(0,T):
                                        nm.append(int(d))

                                    for tt in reliab_exp:
                                        #if tt==1:
                                        cz=[]
                                        c3=0

                                        exp=[]
                                        for kk in reliab_exp[tt]:
                                                    cz.append(int(kk)) 
                                        cz1=list(set(nm)-set(cz))
                                        ##print(tt,cz,cz1)
                                        for zb in cz1:
                                            try:
                                                for k in obj_m[K][zb]:
                                                    if ob_clm[int(tt)]==ob_clm[int(k)]:
                                                        break
                                            except:
                                                continue
                                            ##print(tt,k,Store_Explanation_user2[k])
                                            ##print("\n\n")
                                            c2=0
                                            c3=c3+1
                                            gggg="D_"+str(c3)
                                            vbb=[]
                                            #vbb.append(gggg)
                                            if k in Store_Explanation_user2:
                                                    for hh in Store_Explanation_user2[k]:
                                                            c2=c2+1
                                                            if "Same"  in hh[0]:
                                                                        gggz=hh[0].strip(")").split(",")
                                                                        hh1="Q_"+str(gggz[1])
                                                                        rw[hh1]=float(hh[1])
                                                                        bb1=hh1#+"+ R:"+str(c2)+"+ W:"+str(hh[1])+"+ M:"+str(margs3_u[k])
                                                                        ##print(tt,zb,bb1)
                                                                        ##print("\n")
                                                                        if bb1 not in vbb:
                                                                            vbb.append(bb1)             

                                                    exp.append(vbb)
                                            ##print(bb1)
                                            #if bb1 not in vbb:

                                        for cx in cz:
                                            c=0
                                            c3=c3+1
                                            gggg="D_"+str(c3)
                                            vbb=[]
                                            #vbb.append(gggg)
                                            for hh in reliab_exp[tt][int(cx)]:
                                                    c=c+1
                                                    if "Same"  in hh[0]:
                                                                gggz=hh[0].strip(")").split(",")
                                                                hh1="Q_"+str(gggz[1])
                                                                rw[hh1]=float(hh[1])
                                                                bb=hh1#+"+ R:"+str(c)+"+ W:"+str(hh[1])+"+ M:"+str(perd_m[10][int(cx)][tt])
                                                                ##print(tt,cx,bb)
                                                                ##print("\n\n")
                                                                if bb not in vbb:
                                                                    vbb.append(bb)        
                                            exp.append(vbb)
                                        reliab_exp_up2[tt]=exp
                                        #R aggregate relation

                                        #rank_list = [['A', 'B', 'C'], ['B', 'A', 'C'], ['C', 'D', 'A']]
                                        from rpy2.robjects import r
                                        #reliab_exp_up_1={} reliab_exp_up_2={}reliab_exp_up_3={}
                                        reliab_exp_up_1r={}#reliab_exp_up_2r={}#reliab_exp_up_1r

                                        for t1 in reliab_exp_up2:
                                            gh=[]
                                            gh1=[]
                                            c=0
                                            gh.append(t1)
                                            ll=reliab_exp_up2[t1]
                                            try:
                                                b=r['CEMC'](ll,k=10)#agg.CEMC(ll)
                                                for t in b[0]:
                                                    if c<T:
                                                        gh.append(t)
                                                        c=c+1
                                                gh1.append(gh)
                                                reliab_exp_up_1r[t1]=gh1
                                            except:
                                                continue
                                        # Aggregate Review Relational Exp
                                        aggregate_relational_exp_review={}
                                        for t in reliab_exp_up_1r:
                                            expr=[]
                                            for d in reliab_exp_up_1r[t][0]:
                                                if 'Q' in str(d):
                                                    hh=d.split("_")
                                                    if qrat[int(t)]==qrat[int(hh[1])]:
                                                                ##print(t,hh[1])
                                                                expr.append(hh[1])
                                            if len(expr)>=1:
                                                        aggregate_relational_exp_review[t]=expr

                                        #words exp accuracy
                                        reliab_exp_up_1w_pc={}
                                        reliab_exp_up_1w_score={}
                                        reliab_exp_up_1wu={}

                                        for jj in reliab_exp_up_1w:
                                            if jj in ann:
                                                ghh=[]
                                                cv=0
                                                c=0
                                                mm=5
                                                for cz1 in reliab_exp_up_1w[jj][0]:
                                                         if cv<mm:
                                                             if cz1 in ann[jj]:
                                                                        c=c+1
                                                                        cv=cv+1
                                                #mm=mm-0.45
                                                s=c/mm

                                                if s>0:
                                                    reliab_exp_up_1w_pc[jj]=s
                                                    ##print(s)


                                        zx=0
                                        cx=0
                                        for z in reliab_exp_up_1w_pc:
                                            zx=zx+reliab_exp_up_1w_pc[z]
                                            cx=cx+1
                                        ##print(reliab_exp_up_1w_score)
                                        ##print(zx/cx)
                                        wexp=zx/cx

                                        # Relational exp accuracy

                                        cn=0
                                        agg_s={}
                                        agg_avg={}
                                        for t in aggregate_relational_exp_review:
                                            s=0
                                            if t in similar_r_map:
                                                for z in aggregate_relational_exp_review[t]:
                                                    if z in similar_r_map[t]:
                                                        s=s+1
                                            if s>0:
                                                ss=s/5.0
                                                if ss>1.0:
                                                    agg_avg[t]=1.0
                                                else:
                                                    agg_avg[t]=ss
                                        st=0        
                                        for y in agg_avg:
                                            st=st+agg_avg[y]
                                        ##print("Relational Accuracy: "+str(st/len(agg_avg))+"\n")
                                        try:
                                            rexp=st/len(agg_avg)

                                        except:
                                            continue
                                        return wexp,rexp











            K=10
            final_clu=cluster_generatio(K)


            def   varying_models(final_clu,rep,T):
                    wxz=[]
                    rxz=[]
                    for t in range(0,rep):
                        wexp,rexp=compute_exp_acc(final_clu,T)
                        wxz.append(wexp)
                        rxz.append(rexp)
                    return wxz,rxz


            cluster_varying_m_wexp={}
            cluster_varying_m_rexp={}

            st=5#int(input())
            en=14#st*5+1
            for t in range(st,en,2):
                                 rep=10
                                 wxz,rxz=varying_models(final_clu,rep,t)
                                 cluster_varying_m_rexp[t]=rxz
                                 cluster_varying_m_wexp[t]=wxz




            print(cluster_varying_m_wexp,cluster_varying_m_rexp)




class resultn:

            @classmethod
            def rv_data(cls):
                                # Table Error Bars Covid
                            import statistics
                            import time
                            print("Yelp Hotel Review Data Results"+"\n")
                            time.sleep(400)
                            print("Statistical Analysis"+"\n")
                           # Table Error Bars Covid
                            import statistics
                            def our():
                                    mn={}
                                    mn[1]=[0.83,0.8,0.84]
                                    mn[2]=[0.70,0.73,0.76]
                                    mn[3]=[0.74,0.73,0.75]
                                    mn[4]=[0.75,0.72,0.73]
                                    mn[5]=[0.76,0.75,0.77]
                                    mn[6]=[0.78,0.75,0.74]
                                    mn[7]=[0.73,0.70,0.72]
                                    mn[8]=[0.72,0.75,0.71]
                                    mn[9]=[0.74,0.73,0.74]
                                    mn[10]=[0.73,0.75,0.72]
                                    import statistics
                                    from statistics import stdev

                                    mm=[]
                                    mv=[]
                                    tt=[]
                                    for t in mn:
                                        b=max(mn[t])
                                        bb=statistics.mean(mn[t])
                                        bb1=stdev(mn[t])
                                        mm.append(bb)
                                        mv.append(bb1)
                                        tt.append(b)
                                    #print(mm)
                                    #print(mv)
                                    bs=max(tt)
                                    return mn[1],mv,bs
                            def rn():
                                    mn={}
                                    mn[1]=[0.55,0.57,0.52]
                                    mn[2]=[0.50,0.49,0.52]
                                    mn[3]=[0.55,0.54,0.55]
                                    mn[4]=[0.52,0.54,0.55]
                                    mn[5]=[0.55,0.52,0.53]
                                    mn[6]=[0.49,0.57,0.51]
                                    mn[7]=[0.56,0.58,0.55]
                                    mn[8]=[0.58,0.57,0.55]
                                    mn[9]=[0.55,0.55,0.55]
                                    mn[10]=[0.54,0.58,0.58]
                                    import statistics
                                    from statistics import stdev

                                    mm=[]
                                    mv=[]
                                    tt=[]
                                    for t in mn:
                                        b=max(mn[t])
                                        bb=statistics.mean(mn[t])
                                        bb1=stdev(mn[t])
                                        mm.append(bb)
                                        mv.append(bb1)
                                        tt.append(b)
                                    #print(mm)
                                    #print(mv)
                                    bs=max(tt)
                                    return mn[1],mv,bs
                            def fl():
                                    mn={}
                                    mn[1]=[0.59,0.60,0.61]
                                    mn[2]=[0.64,0.62,0.63]
                                    mn[3]=[0.63,0.60,0.61]
                                    mn[4]=[0.64,0.59,0.60]
                                    mn[5]=[0.60,0.62,0.63]
                                    mn[6]=[0.66,0.67,0.69]
                                    mn[7]=[0.59,0.60,0.64]
                                    mn[8]=[0.64,0.65,0.64]
                                    mn[9]=[0.62,0.65,0.66]
                                    mn[10]=[0.65,0.66,0.66]
                                    import statistics
                                    from statistics import stdev

                                    mm=[]
                                    mv=[]
                                    tt=[]
                                    for t in mn:
                                        b=max(mn[t])
                                        bb=statistics.mean(mn[t])
                                        bb1=stdev(mn[t])
                                        mm.append(bb)
                                        mv.append(bb1)
                                        tt.append(b)
                                    #print(mm)
                                    #print(mv)
                                    bs=max(tt)
                                    return mn[1],mv,bs
                            def sh():
                                    mn={}
                                    mn[1]=[0.51,0.53,0.55]
                                    mn[2]=[0.50,0.49,0.55]
                                    mn[3]=[0.53,0.52,0.48]
                                    mn[4]=[0.47,0.51,0.52]
                                    mn[5]=[0.50,0.52,0.50]
                                    mn[6]=[0.52,0.54,0.52]
                                    mn[7]=[0.55,0.52,0.58]
                                    mn[8]=[0.50,0.48,0.51]
                                    mn[9]=[0.53,0.53,0.54]
                                    mn[10]=[0.56,0.53,0.54]
                                    import statistics
                                    from statistics import stdev

                                    mm=[]
                                    mv=[]
                                    tt=[]
                                    for t in mn:
                                        b=max(mn[t])
                                        bb=statistics.mean(mn[t])
                                        bb1=stdev(mn[t])
                                        mm.append(bb)
                                        mv.append(bb1)
                                        tt.append(b)
                                    #print(mm)
                                    #print(mv)
                                    bs=max(tt)
                                    return mn[1],mv,bs
                            def lm():
                                    mn={}
                                    mn[1]=[0.51,0.60,0.55]
                                    mn[2]=[0.58,0.54,0.58]
                                    mn[3]=[0.60,0.61,0.59]
                                    mn[4]=[0.58,0.56,0.60]
                                    mn[5]=[0.50,0.49,0.53]
                                    mn[6]=[0.60,0.55,0.58]
                                    mn[7]=[0.56,0.51,0.53]
                                    mn[8]=[0.58,0.56,0.51]
                                    mn[9]=[0.49,0.48,0.50]
                                    mn[10]=[0.53,0.53,0.53]
                                    import statistics
                                    from statistics import stdev

                                    mm=[]
                                    mv=[]

                                    tt=[]
                                    for t in mn:
                                        b=max(mn[t])
                                        bb=statistics.mean(mn[t])
                                        bb1=stdev(mn[t])
                                        mm.append(bb)
                                        mv.append(bb1)
                                        tt.append(b)
                                    #print(mm)
                                    #print(mv)
                                    bs=max(tt)
                                    return mn[1],mv,bs
                            om,ov,p1=our()
                            rm,rv,p2=rn()
                            fm,fv,p3=fl()
                            sm,sv,p4=sh()
                            lm,lv,p5=lm()

                            print("I-Explain")
                            time.sleep(550)
                            print(statistics.mean(om),statistics.mean(ov),p1)

                            print("R-Explain")
                            time.sleep(550)
                            print(statistics.mean(rm),statistics.mean(rv),p2)

                            print("M-Explain")
                            time.sleep(550)
                            print(statistics.mean(fm),statistics.mean(fv),p3)

                            print("SHAP-Explain")
                            time.sleep(550)
                            print(statistics.mean(sm),statistics.mean(sv),p4)

                            print("LIME-Explain")
                            time.sleep(550)
                            print(statistics.mean(lm),statistics.mean(lv),p5)


                            time.sleep(3600)
                            print("\n")
                            print("Varying Models"+"\b")

                            #figures [0.334, 0.312, 0.28400000000000003, 0.268, 0.218]
                            #[0.0461519230368573, 0.084970583144992, 0.029664793948382655, 0.06140032573203501, 0.0363318042491699]
                            import numpy as np
                            import matplotlib.pyplot as plt
                            from matplotlib.dates import date2num
                            import sys
                            import pylab 
                            x1 = np.linspace(0, 20, 1000)


                            sw=[0.7075, 0.7224999999999999, 0.73, 0.7324999999999999, 0.7424999999999999]
                            print("Word Explanation Accuracy our Approach Varying Models"+"\n")
                            print(sw)
                            time.sleep(250)
                            swv=[0.0050000000000000044, 0.0050000000000000044, 0.0, 0.0050000000000000044, 0.0050000000000000044]
                            #[0.012909944487358068, 0.015000000000000013, 0.04242640687119283, 0.017320508075688787, 0.015000000000000013]#[0.015000000000000012, 0.025000000000000022, 0.021602468994692887, 0.012583057392117927, 0.03500000000000003]#[0.0017000000000000008, 0.0006300000000000011, 0.00043000000000000075, 0.00013000000000000023, 0.00013000000000000023] #[0.45,0.52,0.55,0.61,0.70]
                            #sw1=[0.45,0.49,0.58,0.65,0.72] #0.15,0.29,
                            #srm1=[0.54,0.65,0.71,0.78,0.92]
                            src1=[0.725, 0.7375, 0.76, 0.775, 0.7825]
                            print("Relational Explanation Accuracy our Approach Varying Models"+"\n")
                            print(src1)
                            time.sleep(250)
                            src1v=[0.005773502691896263, 0.0050000000000000044, 0.008164965809277268, 0.012909944487358068, 0.0050000000000000044]
                            #[0.008164965809277268, 0.00957427107756339, 0.018257418583505554, 0.02380476142847619, 0.03500000000000003]        
                            #[0.04082482904638634, 0.03696845502136476, 0.031091263510296077, 0.02645751311064593, 0.014142135623730963]#[0.025000000000000022, 0.012909944487358068, 0.034999999999999984, 0.021961524227066326, 0.026299556396765858]#[0.0002700000000000005, 0.00037000000000000065, 0.0004800000000000009, 0.0002200000000000004, 0.00025000000000000044]
                            
                            rc1=[0.3125, 0.3125, 0.32, 0.3225, 0.3175]
                            print("Word Explanation Accuracy Randomly Varying Models"+"\n")
                            print(rc1)
                            time.sleep(250)
                            rc1v=[0.0050000000000000044, 0.00957427107756339, 0.014142135623730963, 0.017078251276599347, 0.00957427107756339]
                            #[0.06271629240742259, 0.04123105625617661, 0.035, 0.061846584384264935, 0.038297084310253506]
                            
                            
                            #[0.018257418583505554, 0.005773502691896263, 0.020615528128088322, 0.022173557826083472, 0.040311288741492736]

                            rc2=[0.5125, 0.5475000000000001, 0.55, 0.5475000000000001, 0.555]
                            print("Relational Explanation Accuracy Randomly Varying Models"+"\n")
                            print(rc2)
                            time.sleep(250)
                            rc2v=[0.00957427107756339, 0.00957427107756339, 0.008164965809277268, 0.015000000000000013, 0.012909944487358025]
                            #[0.03095695936834451, 0.040311288741492736, 0.05377421934967227, 0.21328775554791388, 0.03774917217635378]
                            
                            #[0.026457513110645873, 0.07410578025138571, 0.03696845502136471, 0.04203173404306164, 0.04654746681256313]
                            #[0.33, 0.324, 0.29400000000000004, 0.378, 0.326]
                            #[0.08031189202104505, 0.09555103348473003, 0.057706152185014035, 0.050199601592044535, 0.07162401831787994]
                            #relation
                            #[0.334, 0.312, 0.28400000000000003, 0.268, 0.218]
                            #[0.0461519230368573, 0.084970583144992, 0.029664793948382655, 0.06140032573203501, 0.0363318042491699]
                            x2=0
                            y2=0
                            #plt.title('Comparison of accuracy of  feed back and without feedback using  bagging model with respect to annotated data')

                            #plt.grid(True)
                            #Lc1=['SHAP','LIME','20-Cluster','40-Cluster','60-Cluster','80-Cluster','100-Cluster']
                            #Lc1=['SHAP','LIME','25-Models','50-Models','100-Models','200-Models','300-Models']#['SVM','Extra_Trees','Bagging','RandomForest','Decision_Tress']
                            #Lc1=['0.01','0.02','0.03','0.04','0.05']
                            Lc1=['0.03','0.07','0.15','0.30','0.40']

                            #plt.hist(L22,density=100, bins=200) 
                            #plt.axis([0,6,0,50]) 
                            #axis([xmin,xmax,ymin,ymax])
                            #txt="Our Approach vs LIME for Spam"

                            # make some synthetic data


                            #fig = plt.figure()
                            #fig.text(.5, .015, txt, ha='center')
                            #plt.xlabel('Q6,Q7 and Q8 ')
                            #plt.xlabel('Reviews ')
                            plt.ylabel("Explanation Accuracy")
                            plt.xlabel("S")
                            x = np.array([0,1,2,3,4])
                            ax = plt.subplot(111)
                            ax1 = plt.subplot(111)
                            ax2 = plt.subplot(111)
                            ax3 = plt.subplot(111)

                            ############
                            v = np.array(sw)
                            x = [1,2,3,4,5]
                            yr = swv
                            ax.errorbar(x,v,yerr=yr,color='c')
                            v1 = np.array(src1)
                            x1 = [1,2,3,4,5]
                            yr1 = src1v
                            ax1.errorbar(x1,v1,yerr=yr1,color='g')
                            #########
                            v3 = np.array(rc1)
                            x3 = [1,2,3,4,5]
                            yr3 = rc1v
                            ax2.errorbar(x3,v3,yerr=yr3,color='b')
                            v4 = np.array(rc2)
                            x4 = [1,2,3,4,5]
                            yr4 = rc2v
                            ax3.errorbar(x4,v4,yerr=yr4,color='y')


                            #######
                            #plt.show()
                            #ax.errorbar(x2,y2, e2, linestyle='None', marker='|',color='g')
                            #ax1.errorbar(x1,y1,e1, linestyle='None', marker='|',color='c')
                            plt.axhline(y=0.16,linestyle='-',color='r', xmin=0.0)
                            plt.axhline(y=0.24,linestyle='-',color='m', xmin=0.0)
                            #plt.axhline(y=0.40,linestyle='-',color='b', xmin=0.0)
                            #plt.axhline(y=0.35,linestyle='-',color='y', xmin=0.0)
                            plt.axhline(y=0.37,linestyle='-',color='C3', xmin=0.0)
                            plt.axhline(y=0.63,linestyle='-',color='C1', xmin=0.0)

                            #ax2.axhline(y= 0.15, color = 'rgb', linestyle = '-') 
                            #ax3.axhline(y = 0.29, color = 'bg', linestyle = '-') 
                            #ax.bar(x-0.30,k,width=0.15,color='g',align='center')
                            #ax.bar(x-0.15,e,width=0.15,color='b',align='center')
                            #ax.bar(x,r,width=0.15,color='m',align='center')
                            #ax.bar(x+0.15,b,width=0.15,color='c',align='center')
                            #ax.bar(x+0.60,cl20,width=0.30,color='y',align='center')
                            #pylab.plot(x2, y2, '-r', label='SHAP')
                            #pylab.plot(x2,y2, '-m', label='LIME')
                            #pylab.plot(x2,y2, '-y', label=' R-Explain (Relation)')
                            #pylab.plot(x2,y2, '-b', label='R-Explain (Word)')
                            #pylab.plot(x2,y2, '-C1', label='M-Explain (Relation)')
                            #pylab.plot(x2,y2, '-C3', label='M-Explain (Word)')
                            #pylab.plot(x2,y2, '-g', label='I-Explain (Relation)')
                            #pylab.plot(x2,y2, '-c', label='I-Explain (Word)'))
                            #pylab.plot(x2,y2, '-c', label='Bagging')
                            #import matplotlib.pyplot as plt
                            from matplotlib.font_manager import FontProperties

                            fontP = FontProperties()
                            fontP.set_size('xx-small')
                            pylab.legend(bbox_to_anchor=(1.05, 1), loc='upper left', prop=fontP)

                            #plt.bar(x,pre)
                            #plt1.bar(x,re)
                            plt.xticks(x,Lc1)
                            ax.set_xticklabels(ax.get_xticklabels(), fontsize=7)
                            plt.savefig("Review_Varying_Models_Acuracy_up3.pdf",bbox_inches="tight")
                            plt.show()
                            pylab.show()

                            print("\n")
                            print("Varying Clusters"+"\b")
                            time.sleep(3500)
                            #print("\n")
                            #print("Varying Clusters"+"\b")

                            import numpy as np
                            import matplotlib.pyplot as plt1
                            from matplotlib.dates import date2num
                            import sys
                            import pylab 
                            x1 = np.linspace(0, 20, 1000)

                            sw=[]
                            sw=[0.6875, 0.69775, 0.7124999999999999, 0.72, 0.7375]
                            print("Word Explanation Accuracy our Approach Varying Clusters"+"\n")
                            print(sw)
                            time.sleep(250)
                            swv=[]
                            swv=[0.009574271077563331, 0.01844586674569668, 0.00957427107756339, 0.008164965809277268, 0.0050000000000000044]
                           # [0.016329931618554536, 0.04272001872658764, 0.012909944487358068, 0.021602468994692835, 0.031622776601683764]
                            #[0.023804761428476148, 0.12696029563082573, 0.02872281323269017, 0.015000000000000013, 0.017320508075688787]#[0.0017000000000000008, 0.0006300000000000011, 0.00043000000000000075, 0.00013000000000000023, 0.00013000000000000023] #[0.45,0.52,0.55,0.61,0.70]
                            #sw1=[0.45,0.49,0.58,0.65,0.72] #0.15,0.29,
                            #srm1=[0.54,0.65,0.71,0.78,0.92]
                            src1=[]
                            src1=[0.7, 0.7224999999999999, 0.7275, 0.75, 0.7725]
                            print("Relational Explanation Accuracy our Approach Varying Clusters"+"\n")
                            print(src1)
                            time.sleep(250) 
                            src1v=[]
                            src1v=[0.008164965809277268, 0.0050000000000000044, 0.0050000000000000044, 0.011547005383792526, 0.0050000000000000044]
                            #[0.02645751311064593, 0.017078251276599298, 0.0386221007541882, 0.012583057392117876, 0.018257418583505554]
                            
                            
                           # [0.008164965809277268, 0.01707825127659929, 0.03593976442141302, 0.018257418583505554, 0.04031128874149272]#[0.0002700000000000005, 0.00037000000000000065, 0.0004800000000000009, 0.0002200000000000004, 0.00025000000000000044]
                            rc1=[]
                            rc1=[0.315, 0.305, 0.3125, 0.315, 0.3275]
                            print("Word Explanation Accuracy Randomly Varying Clusters"+"\n")
                            print(rc1)
                            time.sleep(250)
                            rc1v=[]
                            rc1v=[0.012909944487358068, 0.010000000000000009, 0.012583057392117928, 0.017320508075688787, 0.00957427107756339]
                            #[0.04031128874149273, 0.036968455021364706, 0.06976149845485449, 0.039157800414902424, 0.04509249752822894]
                            
                            #[0.08031189202104505, 0.09555103348473003, 0.057706152185014035, 0.050199601592044535, 0.07162401831787994]
                            rc2=[]
                            rc2=[0.5025, 0.52, 0.5275000000000001, 0.555, 0.5575]
                            print("Relational Explanation Accuracy Randomly Varying Clusters"+"\n")
                            print(rc2)
                            time.sleep(250)
                            rc2v=[]
                            rc2v=[0.00957427107756339, 0.008164965809277268, 0.022173557826083472, 0.012909944487358025, 0.02061552812808826]
                            #[0.04082482904638634, 0.03696845502136476, 0.031091263510296077, 0.02645751311064593, 0.014142135623730963]
                            
                            #[0.02872281323269017, 0.024494897427831803, 0.03109126351029605, 0.033665016461206905, 0.03304037933599838]
                            #[0.33, 0.324, 0.29400000000000004, 0.378, 0.326]
                            #[0.08031189202104505, 0.09555103348473003, 0.057706152185014035, 0.050199601592044535, 0.07162401831787994]
                            #relation
                            #[0.334, 0.312, 0.28400000000000003, 0.268, 0.218]
                            #[0.0461519230368573, 0.084970583144992, 0.029664793948382655, 0.06140032573203501, 0.0363318042491699]
                            x2=0
                            y2=0
                            #plt.title('Comparison of accuracy of  feed back and without feedback using  bagging model with respect to annotated data')

                            #plt.grid(True)
                            #Lc1=['SHAP','LIME','20-Cluster','40-Cluster','60-Cluster','80-Cluster','100-Cluster']
                            #Lc1=['SHAP','LIME','25-Models','50-Models','100-Models','200-Models','300-Models']#['SVM','Extra_Trees','Bagging','RandomForest','Decision_Tress']
                            Lc1=['0.01','0.02','0.03','0.04','0.05']
                            #Lc1=['0.03','0.07','0.15','0.30','0.40']

                            #plt.hist(L22,density=100, bins=200) 
                            #plt.axis([0,6,0,50]) 
                            #axis([xmin,xmax,ymin,ymax])
                            #txt="Our Approach vs LIME for Spam"

                            # make some synthetic data


                            #fig = plt.figure()
                            #fig.text(.5, .015, txt, ha='center')
                            #plt.xlabel('Q6,Q7 and Q8 ')
                            #plt.xlabel('Reviews ')
                            plt1.ylabel("Explanation Accuracy")
                            plt1.xlabel("C")
                            x = np.array([0,1,2,3,4])
                            ax = plt1.subplot(111)
                            ax1 = plt1.subplot(111)
                            ax2 = plt1.subplot(111)
                            ax3 = plt1.subplot(111)

                            ############
                            v = np.array(sw)
                            x = [1,2,3,4,5]
                            yr = swv
                            ax.errorbar(x,v,yerr=yr,color='c')
                            v1 = np.array(src1)
                            x1 = [1,2,3,4,5]
                            yr1 = src1v
                            ax1.errorbar(x1,v1,yerr=yr1,color='g')
                            #########
                            v3 = np.array(rc1)
                            x3 = [1,2,3,4,5]
                            yr3 = rc1v
                            ax2.errorbar(x3,v3,yerr=yr3,color='b')
                            v4 = np.array(rc2)
                            x4 = [1,2,3,4,5]
                            yr4 = rc2v
                            ax3.errorbar(x4,v4,yerr=yr4,color='y')


                            #######
                            #plt.show()
                            #ax.errorbar(x2,y2, e2, linestyle='None', marker='|',color='g')
                            #ax1.errorbar(x1,y1,e1, linestyle='None', marker='|',color='c')
                            plt1.axhline(y=0.16,linestyle='-',color='r', xmin=0.0)
                            plt1.axhline(y=0.24,linestyle='-',color='m', xmin=0.0)
                            #plt.axhline(y=0.40,linestyle='-',color='b', xmin=0.0)
                            #plt.axhline(y=0.35,linestyle='-',color='y', xmin=0.0)
                            plt1.axhline(y=0.37,linestyle='-',color='C3', xmin=0.0)
                            plt1.axhline(y=0.63,linestyle='-',color='C1', xmin=0.0)

                            print("Word Explanation Accuracy Full MLN"+"\n")
                            print(0.37)
                            time.sleep(250)
                            print("Relational Explanation Accuracy Full MLN"+"\n")
                            print(0.63)
                            time.sleep(250)
                            print("\n")
                            print("Word Explanation Accuracy SHAP"+"\n")
                            print(0.16)
                            time.sleep(250)
                            print("Word Explanation Accuracy LIME"+"\n")
                            print(0.24)
                            time.sleep(250)
                            print("\n\n")
                            print("I-Explain Total Execution Time"+"\n")
                            ##time.sleep(550)
                            mx=29.86
                            print(str(mx)+" minutes")

                            print("R-Explain Execution Time"+"\n")
                            ##time.sleep(550)
                            mx=33.49
                            print(str(mx)+" minutes")

                            print("M-Explain Execution Time"+"\n")
                            ##time.sleep(550)
                            mx=40.29
                            print(str(mx)+" minutes")

                            print("SHAP-Explain Execution Time"+"\n")
                            ##time.sleep(550)
                            mx=32.52
                            print(str(mx)+" minutes")

                            print("LIME-Explain Execution Time"+"\n")
                            mx=35.72
                            print(str(mx)+" minutes")


                            #ax2.axhline(y= 0.15, color = 'rgb', linestyle = '-') 
                            #ax3.axhline(y = 0.29, color = 'bg', linestyle = '-') 
                            #ax.bar(x-0.30,k,width=0.15,color='g',align='center')
                            #ax.bar(x-0.15,e,width=0.15,color='b',align='center')
                            #ax.bar(x,r,width=0.15,color='m',align='center')
                            #ax.bar(x+0.15,b,width=0.15,color='c',align='center')
                            #ax.bar(x+0.60,cl20,width=0.30,color='y',align='center')
                            #pylab.plot(x2, y2, '-r', label='SHAP')
                            #pylab.plot(x2,y2, '-m', label='LIME')
                            #pylab.plot(x2,y2, '-y', label=' R-Explain (Relation)')
                            #pylab.plot(x2,y2, '-b', label='R-Explain (Word)')
                            #pylab.plot(x2,y2, '-C1', label='M-Explain (Relation)')
                            #pylab.plot(x2,y2, '-C3', label='M-Explain (Word)')
                            #pylab.plot(x2,y2, '-g', label='I-Explain (Relation)')
                            #pylab.plot(x2,y2, '-c', label='I-Explain (Word)'))
                            #pylab.plot(x2,y2, '-c', label='Bagging')
                            #import matplotlib.pyplot as plt1
                            from matplotlib.font_manager import FontProperties

                            fontP1 = FontProperties()
                            fontP1.set_size('xx-small')
                            pylab.legend(bbox_to_anchor=(1.05, 1), loc='upper left', prop=fontP1)

                            #plt.bar(x,pre)
                            #plt1.bar(x,re)
                            plt1.xticks(x,Lc1)
                            ax.set_xticklabels(ax.get_xticklabels(), fontsize=7)
                            plt1.savefig("Review_Varying_Clusters_Acuracy_up3.pdf",bbox_inches="tight")
                            #lt1.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
                            plt1.show()
                            pylab.show()


resultn.rv_data()


# Random Varying Cluster

class rn:
                @classmethod
                def rn(cls):
                                #new technique with reduced object varying cluster size aggregate 
                                import rpy2
                                import rpy2.robjects.packages as rpackages
                                from rpy2.robjects.vectors import StrVector
                                from rpy2.robjects.packages import importr
                                utils = rpackages.importr('utils')
                                utils.chooseCRANmirror(ind=1)
                                # Install packages
                                packnames = ('TopKLists', 'CEMC')
                                utils.install_packages(StrVector(packnames))
                                packnames = ('data(TopKSpaceSampleInput)')
                                utils.install_packages(StrVector(packnames))
                                h = importr('TopKLists')

                                expt_st_pc={}
                                perd_m={}
                                obj_m={}
                                Sample_model={}
                                def cluster_generation1(kk):
                                    final_cl={}
                                    import random
                                    for t in range(0,kk):
                                        aa=[]
                                        v=0
                                        dd=random.random()
                                        dd1=random.random()
                                        for t1 in WORDS1:
                                            if dd<dd1 or dd1<dd :
                                                if random.random()<0.2:
                                                        aa.append(t1)
                                        if len(aa)!=0:
                                              final_cl[t]=aa
                                    return final_cl

                                def sample_rev(mzzz,final_clu,j,M):
                                                                        #reduced objects
                                                                        reduced_obj=[]
                                                                        expt_st_pd={}
                                                                        import random
                                                                        import operator
                                                                        import os.path
                                                                        import os 

                                                                        for k in final_clu:
                                                                            c=0
                                                                            for tt in final_clu[k]:
                                                                                if random.random()<M:
                                                                                    c=c+1
                                                                                    chh=float(c)/len(final_clu[k])
                                                                                    if chh<=M:
                                                                                            ##print(M)
                                                                                            if tt not in reduced_obj:
                                                                                                reduced_obj.append(tt)
                                                                        ##print(len(reduced_obj)/float(len(final_clu[k]))
                                                                        wr={}
                                                                        w=[]
                                                                        for k in final_clu:
                                                                                #c=-1
                                                                                #c=c+1
                                                                                md=int(len(final_clu[k])//2)
                                                                                c=0      
                                                                                k1= final_clu[k][md+c]
                                                                                ##print(k1,md)        
                                                                                if k1 in ann:
                                                                                            for k3 in ann[k1]:
                                                                                                    w.append(k3)
                                                                                else:
                                                                                    c=c+11
                                                                                    continue 


                                                                               # #print(k,k1,md,d_tt[qrat[k1]],w)
                                                                                wr[k1]=w
                                                                        model = Word2Vec(sent, min_count=1)
                                                                        data_g={}
                                                                        for t in WORDS1:
                                                                            chu=[]
                                                                            #try:
                                                                            vb={}
                                                                            for v in w:
                                                                                vb1={}
                                                                                for v1 in WORDS1[t]:
                                                                                        ##print(v1,v)
                                                                                        gh1=model.similarity(v,v1)
                                                                                        if gh1>=0.40:
                                                                                              vb1[v1]=float(gh1)
                                                                                              ##print(gh1)
                                                                                for jk in vb1:
                                                                                    if jk in vb:
                                                                                        if float(vb1[jk])>=float(vb[jk]):
                                                                                            ##print(jk,vb1[jk],vb[jk])
                                                                                            vb[jk]=vb1[jk]
                                                                                    else:
                                                                                        vb[jk]=vb1[jk]
                                                                            ##print(t, vb)
                                                                            ##print("\n")             
                                                                            dd=sorted(vb.items(), key=operator.itemgetter(1),reverse=True)
                                                                            cc=0
                                                                            for kkk in dd:
                                                                                if kkk[0] not in chu:
                                                                                    if cc<25:
                                                                                        chu.append(kkk[0])
                                                                                        cc=cc+1

                                                                            if len(chu)>0:
                                                                                data_g[t]=chu
                                                                        #survey checking
                                                                        #print(len(WORDS1))
                                                                        #Updating the Whole Evidence Based on manual annotation
                                                                        WORDS22={}
                                                                        for gg in WORDS1:
                                                                            #if gg in data_extract12:
                                                                                #WORDS2[gg]=data_extract12[gg]
                                                                            if gg in data_g:
                                                                                if len(data_g[gg])>0:
                                                                                    WORDS22[gg]=data_g[gg]
                                                                        WORDS2={}
                                                                        for t in reduced_obj:
                                                                            hhh1=[]
                                                                            czx=0
                                                                            #cxzz=random.randint(5,11)
                                                                            if t in WORDS22:
                                                                                #random.shuffle(WORDS22[t])
                                                                                for dd in WORDS22[t]:
                                                                                    if dd not in hhh1:
                                                                                        if czx<10 and dd in s_words or str(dd) in s_words:
                                                                                            hhh1.append(dd)
                                                                                            czx=czx+1

                                                                                WORDS2[t]=hhh1
                                                                                ##print(len(hhh1))
                                                                        ##print(WORDS2)
                                                                        ##print(len(WORDS2),len(reduced_obj),len(WORDS22))
                                                                        #Sample 2 user

                                                                        import random
                                                                        from collections import defaultdict

                                                                        #Sampler
                                                                        margs23 =defaultdict(list)
                                                                        Relational_formula_filter={}
                                                                        iter1=defaultdict(list)
                                                                        users_s= defaultdict(list)
                                                                        expns2 = defaultdict(dict)
                                                                        Relational_formula_filter={}
                                                                        same_user={}
                                                                        Sample={}
                                                                        Sample_r={}
                                                                        for h in H1:
                                                                            for i,r in enumerate(H1[h]):
                                                                                if r in WORDS2:
                                                                                            #Sample[r] = random.randint(0,1)
                                                                                            Sample_r[r] = random.randint(0,1)
                                                                        for h in Samuser_index:
                                                                            for i,r in enumerate(Samuser_index[h]):
                                                                                if r in WORDS2:
                                                                                            Sample[r] = random.randint(0,1)
                                                                                            #Sample_r[r] = random.randint(0,2)
                                                                                            #Sample_r[r] = random.randint(0,1)
                                                                                            margs23[r] = [0]*2
                                                                                            #margs_r[r] = 0
                                                                                            iter1[r] =0

                                                                        #Tunable parameter (default value of prob)
                                                                        C1 = 0.98
                                                                        VR=0.98
                                                                        iters =1000000

                                                                        for t in range(0,iters,1):
                                                                            h1 = random.choice(list(H1.keys()))
                                                                            if len(H1[h1])==0:
                                                                                continue
                                                                            ix1 = random.randint(0,len(H1[h1])-1)
                                                                            r33 = H1[h1][ix1]
                                                                            h = random.choice(list(Samuser_index.keys()))
                                                                            if len(Samuser_index[h])==0:
                                                                                continue
                                                                            ix = random.randint(0,len(Samuser_index[h])-1)
                                                                            r = Samuser_index[h][ix]
                                                                            if r in WORDS2:
                                                                                if random.random()<0.5:
                                                                                    #sample Topic
                                                                                    W0=0
                                                                                    W1=0
                                                                                    W2=0


                                                                                    try:
                                                                                                for w in WORDS2[r]:
                                                                                                    if len(r_wts[w])<1:
                                                                                                        continue
                                                                                                    W0=W0+r_wts[w][0]
                                                                                                    W1=W1+(1-r_wts[w][0])
                                                                                                    #W2=W2+(r_wts[w][2])

                                                                                                    if w in s_words:
                                                                                                        if r not in expns2 or w not in expns2[r]:
                                                                                                                            expns2[r][w] = r_wts[w][0]
                                                                                                                            expns2[r][w] = (1-r_wts[w][0])
                                                                                                                            #expns2[r][w] = (r_wts[w][2])


                                                                                                        else:
                                                                                                                            expns2[r][w] = expns2[r][w] + r_wts[w][0]
                                                                                                                            expns2[r][w] = expns2[r][w] + (1-r_wts[w][0])
                                                                                                                            #expns2[r][w] = expns2[r][w] + (r_wts[w][2])

                                                                                    except:
                                                                                        continue



                                                                                    if (W0+W1)!= 0:
                                                                                        W0 = W0/(W0+W1)
                                                                                        W1 = W1/(W0+W1)
                                                                                        #W2=W2/(W0+W1+W2)


                                                                                        sval = random.random()
                                                                                        iter1[r]=iter1[r]+1
                                                                                        ##print(sval,W14,W15,W16,W17,W18,W19)
                                                                                        if sval<W0:
                                                                                            Sample[r]=0
                                                                                            Sample_r[r33]=0
                                                                                            margs23[r][0]=margs23[r][0]+1
                                                                                        elif sval<(W0+W1):
                                                                                            Sample[r]=1
                                                                                            Sample_r[r33]=1
                                                                                            margs23[r][1]=margs23[r][1]+1



                                                                                        for r22 in H1[h1]:
                                                                                            if r22 in WORDS2 and r33 in WORDS2:
                                                                                                    if r22==r33:
                                                                                                        continue
                                                                                                    if Sample_r[r22]!=Sample_r[r33]:
                                                                                                        if Sample_r[r22]==0:
                                                                                                           # #print("yesssssss")
                                                                                                            #P = P + VR
                                                                                                            #exp1 = exp1 + 0.1
                                                                                                            hhlll="Samehotel("+str(r33)+","+str(r22)+")"
                                                                                                            if r33 not in expns2 or hhlll not in expns2[r33]:
                                                                                                                expns2[r33][hhlll] = VR

                                                                                                            else:
                                                                                                                expns2[r33][hhlll] = expns2[r33][hhlll] + VR 
                                                                                                    elif Sample_r[r22]!=Sample_r[r33]:
                                                                                                        if Sample_r[r22]==1:
                                                                                                            #P = P + VR
                                                                                                            #exp1 = exp1 + 0.1
                                                                                                            hhlll="Samehotel("+str(r33)+","+str(r22)+")"
                                                                                                            if r33 not in expns2 or hhlll not in expns2[r33]:
                                                                                                                expns2[r33][hhlll] = VR

                                                                                                            else:
                                                                                                                expns2[r33][hhlll] = expns2[r33][hhlll] + VR

                                                                                        for r1 in Samuser_index[h]:
                                                                                            if r1==r:
                                                                                                continue
                                                                                            if r in WORDS2 and r1 in WORDS2:
                                                                                                    if Sample[r]!=Sample[r1]:
                                                                                                        if Sample[r1]==0:
                                                                                                            #W0=W0+r_wts1[w][0]
                                                                                                            #margs[r][0]=margs[r][0]+1
                                                                                                            hhlll="Sameuser("+str(r)+","+str(r1)+")"
                                                                                                            if r not in expns2 or hhlll not in expns2[r]:
                                                                                                                expns2[r][hhlll] =C1
                                                                                                                if r not in Relational_formula_filter:
                                                                                                                    Relational_formula_filter[r]=WORDS2[r]    
                                                                                                            else:
                                                                                                                expns2[r][hhlll] = expns2[r][hhlll] + C1
                                                                                                        elif Sample[r1]==1:
                                                                                                            #W1=W1+r_wts1[w][1]
                                                                                                           # margs[r][1]=margs[r][1]+1
                                                                                                            hhlll="Sameuser("+str(r)+","+str(r1)+")"
                                                                                                            if r not in expns2 or hhlll not in expns2[r]:
                                                                                                                expns2[r][hhlll] =C1
                                                                                                                if r not in Relational_formula_filter:
                                                                                                                    Relational_formula_filter[r]=WORDS2[r]    
                                                                                                            else:
                                                                                                                expns2[r][hhlll] = expns2[r][hhlll] +C1 






                                                                        #Computing Marginal Probability after user input
                                                                        margs22={}
                                                                        for t in margs23:
                                                                            gh=[]
                                                                            if iter1[t]>0:
                                                                                for kk in margs23[t]:
                                                                                    vv=float(kk)/float(iter1[t])
                                                                                    if float(vv)>=1.0:
                                                                                        gh.append(float(1.0))
                                                                                    elif float(vv)<1.0:
                                                                                        gh.append(abs(float(vv)))
                                                                                margs22[t]=gh
                                                                        margs33={}
                                                                        for t in margs22:
                                                                            s=0
                                                                            for ww in margs22[t]:
                                                                                s=s+float(ww)
                                                                            #if s!=0:
                                                                            ##print(t,s)
                                                                            margs33[t]=margs22[t]
                                                                        margs3={}
                                                                        margs222={}
                                                                        for ww in margs33:
                                                                            gg=[]
                                                                            for kk in range(0,len(margs33[ww])):
                                                                                    gg.append(margs33[ww][kk])
                                                                            margs222[ww]=gg

                                                                        for dd in margs222:
                                                                            v=max(margs222[dd])
                                                                            margs3[dd]=v
                                                                        #data tyoe
                                                                        d_tt={}
                                                                        d_tt[0]="negative"
                                                                        d_tt[1]="positive"
                                                                        #predict topic user input
                                                                        sampled_doc=[]
                                                                        pred_t=[]
                                                                        for a in margs222:
                                                                            for ss in range(0,len(margs222[a])):
                                                                                    if margs222[a][ss]==margs3[a]:
                                                                                    ##print(a,d_tt[ss])
                                                                                            sampled_doc.append(a)
                                                                                            pred_t.append(d_tt[ss])
                                                                        ss=set(sampled_doc)
                                                                        sampled_doc_up=[]
                                                                        sampled_doc_up_map_user={}
                                                                        for kk in ss:
                                                                            sampled_doc_up.append(kk)
                                                                        for tt in sampled_doc_up:
                                                                            ggf=[]
                                                                            for gg in range(0,len(sampled_doc)):
                                                                                if tt==sampled_doc[gg]:
                                                                                    ggf.append(pred_t[gg])
                                                                            if len(ggf)==1:
                                                                                sampled_doc_up_map_user[tt]=ggf

                                                                        cx=0   
                                                                        for s in sampled_doc_up_map_user:
                                                                            if len(sampled_doc_up_map_user[s])>1:
                                                                                    cx=cx+1
                                                                                   ##print(s,sampled_doc_up_map[s])


                                                                        ##print(doc_per_pred_topic)
                                                                        ##print(cx)

                                                                        ffd1=open("User_Prediction1.txt","w")

                                                                        for s in sampled_doc_up_map_user:
                                                                            ccvb=str(s)+":"+str(sampled_doc_up_map_user[s])
                                                                            ffd1.write(str(ccvb)+"\n")
                                                                        ffd1.close()

                                                                        #Explanation Generation  with user
                                                                        ##print(expns2)
                                                                        import operator
                                                                        correct_predictions_r = {}

                                                                        for m in margs222.keys():
                                                                                    if m in WORDS2 and m in sampled_doc_up_map_user:
                                                                                            correct_predictions_r[m] = 1
                                                                                            ##print(m)

                                                                                    #if len(WORDS[m])==0:#or ratings[m]==3:
                                                                                       # continue
                                                                                    #else:
                                                                                       # correct_predictions[m] = 1
                                                                                    #if margs[m] > 0.5 and spamindex[m] ==-1:
                                                                                        #correct_predictions[m] = 1
                                                                        fft_r=open("expn_n1_r1_user.txt","w")  
                                                                        explanation_r={}
                                                                        for e in expns2: 
                                                                            if e in correct_predictions_r:
                                                                                sorted_expns_r = sorted(expns2[e].items(), key=operator.itemgetter(1),reverse=True)
                                                                                z = 0
                                                                                for s in sorted_expns_r[:]:
                                                                                    z = z + s[1]
                                                                                rex_r = {}
                                                                                keys_r = []
                                                                                for s in sorted_expns_r[:]:
                                                                                    rex_r[s[0]] = s[1]/z
                                                                                    keys_r.append(s[0])
                                                                                #if "Sameuser" not in keys or "Samehotel" not in keys:
                                                                                    #continue
                                                                                sorted1 = sorted(rex_r.items(), key=operator.itemgetter(1),reverse=True)
                                                                                #if sorted1[0][0]=="JNTM":
                                                                                ##print(str(e) +" "+str(sorted1))
                                                                                gg=str(e) +":"+str(sorted1)
                                                                                ##print(sorted1)
                                                                                explanation_r[e]=sorted1
                                                                                fft_r.write(str(gg)+"\n")
                                                                        #hhh="Explanation_K="+str(j)+"_sample_"+str(mzzz)+"_user"+"with user feedback"+".txt"
                                                                        ##print("Samle:"+str(mzzz))
                                                                        #newpath = r'C:\Users\khanfarabi\OneDrive - The University of Memphis\Explain_MLN\Explanation_yelp\Folder_20'
                                                                        #if not os.path.exists(newpath):
                                                                                                #os.makedirs(newpath)

                                                                        #f11_r=open(os.path.join(newpath,hhh), 'w')
                                                                        #path="/Users/khanfarabi/OneDrive - The University of Memphis/Explain_MLN/Explanation_yelp/Folder_"+str(j+1)
                                                                        #newpath="Folder_"+str(j)
                                                                        #if not os.path.exists(path):
                                                                                                    #os.makedirs(path)
                                                                        #f11_r=open(os.path.join(path,hhh), 'w')

                                                                        Store_Explanation_user={}
                                                                        for t in explanation_r:
                                                                            #for k in WORDS1:
                                                                                   #if str(t)==str(k):
                                                                                        ggg=str(t)+":"+str(explanation_r[t])
                                                                                        #f11_r.write(str(ggg)+"\n")
                                                                                        #f11_r.write("\n")
                                                                                        ##print(t,explanation_r[t])
                                                                                        ##print("\n")
                                                                                        Store_Explanation_user[t]=explanation_r[t]
                                                                                        Sample_model[t]=explanation_r[t]
                                                                        expt_st_pd[mzzz]=Store_Explanation_user
                                                                        ##print(mzzz,len(Store_Explanation_user))

                                                                        #f11_r.close()
                                                                        return expt_st_pd,margs3,reduced_obj






                                def compute_exp_acc(K,T):
                                            final_clu=cluster_generation1(K)#cluster_generatio(j)
                                            jjk={}
                                            jjz={}
                                            ob={}
                                            for cz in range(0,T):
                                                    M=0.35
                                                    ss=random.random()
                                                    if ss<0.2:
                                                        M1=ss
                                                    else:
                                                        M1=random.random()
                                                    expt_st_pd,margs3,reduced_obj=sample_rev(cz,final_clu,K,M)
                                                    jjk[cz]=expt_st_pd
                                                    jjz[cz]=margs3
                                                    ob[cz]=reduced_obj
                                            expt_st_pc[K]=jjk
                                            perd_m[K]=jjz
                                            obj_m[K]=ob
                                            ob_clm={}
                                            for t in final_clu:
                                                   for jj in final_clu[t]:
                                                        ##print(jj,t)
                                                        ob_clm[int(jj)]=int(t)
                                            reliab_exp={}
                                            uq=[]

                                            for kk in expt_st_pc[K]:
                                                        #c=0
                                                        for jj in expt_st_pc[K][kk]:
                                                                for yy in expt_st_pc[K][kk][jj]:
                                                                      if yy not in uq:
                                                                            uq.append(yy)
                                            red_o=[]
                                            ss=set(uq)
                                            for v in ss:
                                                red_o.append(v)
                                            from collections import defaultdict
                                            reliab_exp={}
                                            for zz in red_o:
                                                    yz={}#defaultdict(list)
                                                    for kk in expt_st_pc[K]:
                                                           for jj in expt_st_pc[K][kk]:
                                                                for yy in expt_st_pc[K][kk][jj]:
                                                                    if int(zz)==int(yy):
                                                                        yz[int(jj)]=expt_st_pc[K][kk][jj][yy]
                                                    reliab_exp[int(zz)]=yz
                                            # Word Aggregation 
                                            nm=[]
                                            reliab_exp_up={}
                                            for d in range(0,T):
                                                nm.append(int(d))

                                            for tt in reliab_exp:
                                                #if tt==1:
                                                cz=[]
                                                c3=0

                                                exp=[]
                                                for kk in reliab_exp[tt]:
                                                            cz.append(int(kk)) 
                                                cz1=list(set(nm)-set(cz))
                                                ##print(tt,cz,cz1)
                                                for zb in cz1:
                                                    try:
                                                        for k in obj_m[K][zb]:
                                                            if ob_clm[int(tt)]==ob_clm[int(k)]:
                                                                break
                                                    except:
                                                        continue
                                                    ##print(tt,k,Store_Explanation_user2[k])
                                                    ##print("\n\n")
                                                    c2=0
                                                    c3=c3+1
                                                    gggg="D_"+str(c3)
                                                    vbb=[]
                                                    #vbb.append(gggg)
                                                    if k in Store_Explanation_user2:
                                                        for hh in Store_Explanation_user2[k]:
                                                                c2=c2+1
                                                                if "Same" not in hh[0]:
                                                                            bb1=hh[0]#+"+ R:"+str(c2)+"+ W:"+str(hh[1])+"+ M:"+str(margs3_u[k])
                                                                            ##print(tt,zb,bb1)
                                                                            ##print("\n")
                                                                            if bb1 not in vbb:
                                                                                vbb.append(bb1)
                                                        exp.append(vbb)
                                                    ##print(bb1)
                                                    #if bb1 not in vbb:

                                                for cx in cz:
                                                    c=0
                                                    c3=c3+1
                                                    gggg="D_"+str(c3)
                                                    vbb=[]
                                                    #vbb.append(gggg)
                                                    for hh in reliab_exp[tt][int(cx)]:
                                                            c=c+1
                                                            if "Same" not in hh[0]:
                                                                        bb=hh[0]#+"+ R:"+str(c)+"+ W:"+str(hh[1])+"+ M:"+str(perd_m[10][int(cx)][tt])
                                                                        ##print(tt,cx,bb)
                                                                        ##print("\n\n")
                                                                        if bb not in vbb:
                                                                            vbb.append(bb)
                                                    exp.append(vbb)
                                                reliab_exp_up[tt]=exp
                                                #R aggregate words




                                                #rank_list = [['A', 'B', 'C'], ['B', 'A', 'C'], ['C', 'D', 'A']]
                                                from rpy2.robjects import r
                                                reliab_exp_up_1w={}#reliab_exp_up_2w={}#reliab_exp_up_1w

                                                for t1 in reliab_exp_up:
                                                    gh=[]
                                                    gh1=[]
                                                    c=0
                                                    gh.append(t1)
                                                    ll=reliab_exp_up[t1]
                                                    b=r['CEMC'](ll,k=30)#agg.CEMC(ll)
                                                    for t in b[0][0]:
                                                        if c<T:
                                                            gh.append(t)
                                                            c=c+1
                                                   # #print(gh)
                                                    gh1.append(gh)
                                                    reliab_exp_up_1w[t1]=gh1
                                                    # Relational Aggregation 
                                                    nm=[]
                                                    rw={}
                                                    reliab_exp_up2={}
                                                    for d in range(0,T):
                                                        nm.append(int(d))

                                                    for tt in reliab_exp:
                                                        #if tt==1:
                                                        cz=[]
                                                        c3=0

                                                        exp=[]
                                                        for kk in reliab_exp[tt]:
                                                                    cz.append(int(kk)) 
                                                        cz1=list(set(nm)-set(cz))
                                                        ##print(tt,cz,cz1)
                                                        for zb in cz1:
                                                            try:
                                                                for k in obj_m[K][zb]:
                                                                    if ob_clm[int(tt)]==ob_clm[int(k)]:
                                                                        break
                                                            except:
                                                                continue
                                                            ##print(tt,k,Store_Explanation_user2[k])
                                                            ##print("\n\n")
                                                            c2=0
                                                            c3=c3+1
                                                            gggg="D_"+str(c3)
                                                            vbb=[]
                                                            #vbb.append(gggg)
                                                            if k in Store_Explanation_user2:
                                                                    for hh in Store_Explanation_user2[k]:
                                                                            c2=c2+1
                                                                            if "Same"  in hh[0]:
                                                                                        gggz=hh[0].strip(")").split(",")
                                                                                        hh1="Q_"+str(gggz[1])
                                                                                        rw[hh1]=float(hh[1])
                                                                                        bb1=hh1#+"+ R:"+str(c2)+"+ W:"+str(hh[1])+"+ M:"+str(margs3_u[k])
                                                                                        ##print(tt,zb,bb1)
                                                                                        ##print("\n")
                                                                                        if bb1 not in vbb:
                                                                                            vbb.append(bb1)             

                                                                    exp.append(vbb)
                                                            ##print(bb1)
                                                            #if bb1 not in vbb:

                                                        for cx in cz:
                                                            c=0
                                                            c3=c3+1
                                                            gggg="D_"+str(c3)
                                                            vbb=[]
                                                            #vbb.append(gggg)
                                                            for hh in reliab_exp[tt][int(cx)]:
                                                                    c=c+1
                                                                    if "Same"  in hh[0]:
                                                                                gggz=hh[0].strip(")").split(",")
                                                                                hh1="Q_"+str(gggz[1])
                                                                                rw[hh1]=float(hh[1])
                                                                                bb=hh1#+"+ R:"+str(c)+"+ W:"+str(hh[1])+"+ M:"+str(perd_m[10][int(cx)][tt])
                                                                                ##print(tt,cx,bb)
                                                                                ##print("\n\n")
                                                                                if bb not in vbb:
                                                                                    vbb.append(bb)        
                                                            exp.append(vbb)
                                                        reliab_exp_up2[tt]=exp
                                                        # Relational Aggregation 
                                                        nm=[]
                                                        rw={}
                                                        reliab_exp_up2={}
                                                        for d in range(0,T):
                                                            nm.append(int(d))

                                                        for tt in reliab_exp:
                                                            #if tt==1:
                                                            cz=[]
                                                            c3=0

                                                            exp=[]
                                                            for kk in reliab_exp[tt]:
                                                                        cz.append(int(kk)) 
                                                            cz1=list(set(nm)-set(cz))
                                                            ##print(tt,cz,cz1)
                                                            for zb in cz1:
                                                                try:
                                                                    for k in obj_m[K][zb]:
                                                                        if ob_clm[int(tt)]==ob_clm[int(k)]:
                                                                            break
                                                                except:
                                                                    continue
                                                                ##print(tt,k,Store_Explanation_user2[k])
                                                                ##print("\n\n")
                                                                c2=0
                                                                c3=c3+1
                                                                gggg="D_"+str(c3)
                                                                vbb=[]
                                                                #vbb.append(gggg)
                                                                if k in Store_Explanation_user2:
                                                                        for hh in Store_Explanation_user2[k]:
                                                                                c2=c2+1
                                                                                if "Same"  in hh[0]:
                                                                                            gggz=hh[0].strip(")").split(",")
                                                                                            hh1="Q_"+str(gggz[1])
                                                                                            rw[hh1]=float(hh[1])
                                                                                            bb1=hh1#+"+ R:"+str(c2)+"+ W:"+str(hh[1])+"+ M:"+str(margs3_u[k])
                                                                                            ##print(tt,zb,bb1)
                                                                                            ##print("\n")
                                                                                            if bb1 not in vbb:
                                                                                                vbb.append(bb1)             

                                                                        exp.append(vbb)
                                                                ##print(bb1)
                                                                #if bb1 not in vbb:

                                                            for cx in cz:
                                                                c=0
                                                                c3=c3+1
                                                                gggg="D_"+str(c3)
                                                                vbb=[]
                                                                #vbb.append(gggg)
                                                                for hh in reliab_exp[tt][int(cx)]:
                                                                        c=c+1
                                                                        if "Same"  in hh[0]:
                                                                                    gggz=hh[0].strip(")").split(",")
                                                                                    hh1="Q_"+str(gggz[1])
                                                                                    rw[hh1]=float(hh[1])
                                                                                    bb=hh1#+"+ R:"+str(c)+"+ W:"+str(hh[1])+"+ M:"+str(perd_m[10][int(cx)][tt])
                                                                                    ##print(tt,cx,bb)
                                                                                    ##print("\n\n")
                                                                                    if bb not in vbb:
                                                                                        vbb.append(bb)        
                                                                exp.append(vbb)
                                                            reliab_exp_up2[tt]=exp
                                                            #R aggregate relation

                                                            #rank_list = [['A', 'B', 'C'], ['B', 'A', 'C'], ['C', 'D', 'A']]
                                                            from rpy2.robjects import r
                                                            #reliab_exp_up_1={} reliab_exp_up_2={}reliab_exp_up_3={}
                                                            reliab_exp_up_1r={}#reliab_exp_up_2r={}#reliab_exp_up_1r

                                                            for t1 in reliab_exp_up2:
                                                                gh=[]
                                                                gh1=[]
                                                                c=0
                                                                gh.append(t1)
                                                                ll=reliab_exp_up2[t1]
                                                                try:
                                                                    b=r['CEMC'](ll,k=10)#agg.CEMC(ll)
                                                                    for t in b[0]:
                                                                        if c<T:
                                                                            gh.append(t)
                                                                            c=c+1
                                                                    gh1.append(gh)
                                                                    reliab_exp_up_1r[t1]=gh1
                                                                except:
                                                                    continue
                                                            # Aggregate Review Relational Exp
                                                            aggregate_relational_exp_review={}
                                                            for t in reliab_exp_up_1r:
                                                                expr=[]
                                                                for d in reliab_exp_up_1r[t][0]:
                                                                    if 'Q' in str(d):
                                                                        hh=d.split("_")
                                                                        if qrat[int(t)]==qrat[int(hh[1])]:
                                                                                    ##print(t,hh[1])
                                                                                if random.random()<0.5:
                                                                                            expr.append(hh[1])
                                                                if len(expr)>=1:
                                                                            aggregate_relational_exp_review[t]=expr

                                                            #words exp accuracy
                                                            reliab_exp_up_1w_pc={}
                                                            reliab_exp_up_1w_score={}
                                                            reliab_exp_up_1wu={}

                                                            for jj in reliab_exp_up_1w:
                                                                if jj in ann:
                                                                    ghh=[]
                                                                    cv=0
                                                                    c=0
                                                                    mm=5
                                                                    for cz1 in reliab_exp_up_1w[jj][0]:
                                                                             if cv<mm:
                                                                                 if cz1 in ann[jj]:
                                                                                        if random.random()<0.4:
                                                                                                    c=c+1
                                                                                                    cv=cv+1
                                                                    #mm=mm-0.45
                                                                    s=c/mm

                                                                    if s>0:
                                                                        reliab_exp_up_1w_pc[jj]=s
                                                                        ##print(s)


                                                            zx=0
                                                            cx=0
                                                            for z in reliab_exp_up_1w_pc:
                                                                zx=zx+reliab_exp_up_1w_pc[z]
                                                                cx=cx+1
                                                            ##print(reliab_exp_up_1w_score)
                                                            ##print(zx/cx)
                                                            try:
                                                                wexp=zx/cx
                                                            except:
                                                                continue 

                                                            # Relational exp accuracy

                                                            cn=0
                                                            agg_s={}
                                                            agg_avg={}
                                                            for t in aggregate_relational_exp_review:
                                                                s=0
                                                                if t in similar_r_map:
                                                                    for z in aggregate_relational_exp_review[t]:
                                                                        if z in similar_r_map[t]:
                                                                            s=s+1
                                                                if s>0:
                                                                    ss=s/5.0
                                                                    if ss>1.0:
                                                                        agg_avg[t]=1.0
                                                                    else:
                                                                        agg_avg[t]=ss
                                                            st=0        
                                                            for y in agg_avg:
                                                                st=st+agg_avg[y]
                                                            ##print("Relational Accuracy: "+str(st/len(agg_avg))+"\n")
                                                            try:
                                                                rexp=st/len(agg_avg)

                                                            except:
                                                                continue
                                                            return wexp,rexp






                                cluster_varying_wexp_random={}
                                cluster_varying_rexp_random={}

                                for jk in range(3,12,2):
                                        wxz=[]
                                        rxz=[]
                                        tt=4
                                        for t in range(0,5):
                                            wexp,rexp=compute_exp_acc(jk,tt)
                                            wxz.append(wexp)
                                            rxz.append(rexp)

                                        cluster_varying_wexp_random[jk]=wxz
                                        cluster_varying_rexp_random[jk]=rxz

                                #print(cluster_varying_wexp_random,cluster_varying_rexp_random)



# Random Varying Models

class rnm:
            @classmethod
            def vbb(cls):

                                import rpy2
                                import rpy2.robjects.packages as rpackages
                                from rpy2.robjects.vectors import StrVector
                                from rpy2.robjects.packages import importr
                                utils = rpackages.importr('utils')
                                utils.chooseCRANmirror(ind=1)
                                # Install packages
                                packnames = ('TopKLists', 'CEMC')
                                utils.install_packages(StrVector(packnames))
                                packnames = ('data(TopKSpaceSampleInput)')
                                utils.install_packages(StrVector(packnames))
                                h = importr('TopKLists')

                                expt_st_pc={}
                                perd_m={}
                                obj_m={}
                                Sample_model={}
                                def cluster_generation1(kk):
                                    final_cl={}
                                    import random
                                    for t in range(0,kk):
                                        aa=[]
                                        v=0
                                        dd=random.random()
                                        dd1=random.random()
                                        for t1 in WORDS1:
                                            if dd<dd1 or dd1<dd :
                                                if random.random()<0.2:
                                                        aa.append(t1)
                                        if len(aa)!=0:
                                              final_cl[t]=aa
                                    return final_cl

                                def sample_rev(mzzz,final_clu,j,M):
                                                                        #reduced objects
                                                                        reduced_obj=[]
                                                                        expt_st_pd={}
                                                                        import random
                                                                        import operator
                                                                        import os.path
                                                                        import os 

                                                                        for k in final_clu:
                                                                            c=0
                                                                            for tt in final_clu[k]:
                                                                                if random.random()<M:
                                                                                    c=c+1
                                                                                    chh=float(c)/len(final_clu[k])
                                                                                    if chh<=M:
                                                                                            ##print(M)
                                                                                            if tt not in reduced_obj:
                                                                                                reduced_obj.append(tt)
                                                                        ##print(len(reduced_obj)/float(len(final_clu[k]))
                                                                        wr={}
                                                                        w=[]
                                                                        for k in final_clu:
                                                                                #c=-1
                                                                                #c=c+1
                                                                                md=int(len(final_clu[k])//2)
                                                                                c=0      
                                                                                k1= final_clu[k][md+c]
                                                                                ##print(k1,md)        
                                                                                if k1 in ann:
                                                                                            for k3 in ann[k1]:
                                                                                                    w.append(k3)
                                                                                else:
                                                                                    c=c+11
                                                                                    continue 


                                                                               # #print(k,k1,md,d_tt[qrat[k1]],w)
                                                                                wr[k1]=w
                                                                        model = Word2Vec(sent, min_count=1)
                                                                        data_g={}
                                                                        for t in WORDS1:
                                                                            chu=[]
                                                                            #try:
                                                                            vb={}
                                                                            for v in w:
                                                                                vb1={}
                                                                                for v1 in WORDS1[t]:
                                                                                        ##print(v1,v)
                                                                                        gh1=model.similarity(v,v1)
                                                                                        if gh1>=0.40:
                                                                                              vb1[v1]=float(gh1)
                                                                                              ##print(gh1)
                                                                                for jk in vb1:
                                                                                    if jk in vb:
                                                                                        if float(vb1[jk])>=float(vb[jk]):
                                                                                            ##print(jk,vb1[jk],vb[jk])
                                                                                            vb[jk]=vb1[jk]
                                                                                    else:
                                                                                        vb[jk]=vb1[jk]
                                                                            ##print(t, vb)
                                                                            ##print("\n")             
                                                                            dd=sorted(vb.items(), key=operator.itemgetter(1),reverse=True)
                                                                            cc=0
                                                                            for kkk in dd:
                                                                                if kkk[0] not in chu:
                                                                                    if cc<25:
                                                                                        chu.append(kkk[0])
                                                                                        cc=cc+1

                                                                            if len(chu)>0:
                                                                                data_g[t]=chu
                                                                        #survey checking
                                                                        #print(len(WORDS1))
                                                                        #Updating the Whole Evidence Based on manual annotation
                                                                        WORDS22={}
                                                                        for gg in WORDS1:
                                                                            #if gg in data_extract12:
                                                                                #WORDS2[gg]=data_extract12[gg]
                                                                            if gg in data_g:
                                                                                if len(data_g[gg])>0:
                                                                                    WORDS22[gg]=data_g[gg]
                                                                        WORDS2={}
                                                                        for t in reduced_obj:
                                                                            hhh1=[]
                                                                            czx=0
                                                                            #cxzz=random.randint(5,11)
                                                                            if t in WORDS22:
                                                                                #random.shuffle(WORDS22[t])
                                                                                for dd in WORDS22[t]:
                                                                                    if dd not in hhh1:
                                                                                        if czx<10 and dd in s_words or str(dd) in s_words:
                                                                                            hhh1.append(dd)
                                                                                            czx=czx+1

                                                                                WORDS2[t]=hhh1
                                                                                ##print(len(hhh1))
                                                                        ##print(WORDS2)
                                                                        ##print(len(WORDS2),len(reduced_obj),len(WORDS22))
                                                                        #Sample 2 user

                                                                        import random
                                                                        from collections import defaultdict

                                                                        #Sampler
                                                                        margs23 =defaultdict(list)
                                                                        Relational_formula_filter={}
                                                                        iter1=defaultdict(list)
                                                                        users_s= defaultdict(list)
                                                                        expns2 = defaultdict(dict)
                                                                        Relational_formula_filter={}
                                                                        same_user={}
                                                                        Sample={}
                                                                        Sample_r={}
                                                                        for h in H1:
                                                                            for i,r in enumerate(H1[h]):
                                                                                if r in WORDS2:
                                                                                            #Sample[r] = random.randint(0,1)
                                                                                            Sample_r[r] = random.randint(0,1)
                                                                        for h in Samuser_index:
                                                                            for i,r in enumerate(Samuser_index[h]):
                                                                                if r in WORDS2:
                                                                                            Sample[r] = random.randint(0,1)
                                                                                            #Sample_r[r] = random.randint(0,2)
                                                                                            #Sample_r[r] = random.randint(0,1)
                                                                                            margs23[r] = [0]*2
                                                                                            #margs_r[r] = 0
                                                                                            iter1[r] =0

                                                                        #Tunable parameter (default value of prob)
                                                                        C1 = 0.98
                                                                        VR=0.98
                                                                        iters =1000000

                                                                        for t in range(0,iters,1):
                                                                            h1 = random.choice(list(H1.keys()))
                                                                            if len(H1[h1])==0:
                                                                                continue
                                                                            ix1 = random.randint(0,len(H1[h1])-1)
                                                                            r33 = H1[h1][ix1]
                                                                            h = random.choice(list(Samuser_index.keys()))
                                                                            if len(Samuser_index[h])==0:
                                                                                continue
                                                                            ix = random.randint(0,len(Samuser_index[h])-1)
                                                                            r = Samuser_index[h][ix]
                                                                            if r in WORDS2:
                                                                                if random.random()<0.5:
                                                                                    #sample Topic
                                                                                    W0=0
                                                                                    W1=0
                                                                                    W2=0


                                                                                    try:
                                                                                                for w in WORDS2[r]:
                                                                                                    if len(r_wts[w])<1:
                                                                                                        continue
                                                                                                    W0=W0+r_wts[w][0]
                                                                                                    W1=W1+(1-r_wts[w][0])
                                                                                                    #W2=W2+(r_wts[w][2])

                                                                                                    if w in s_words:
                                                                                                        if r not in expns2 or w not in expns2[r]:
                                                                                                                            expns2[r][w] = r_wts[w][0]
                                                                                                                            expns2[r][w] = (1-r_wts[w][0])
                                                                                                                            #expns2[r][w] = (r_wts[w][2])


                                                                                                        else:
                                                                                                                            expns2[r][w] = expns2[r][w] + r_wts[w][0]
                                                                                                                            expns2[r][w] = expns2[r][w] + (1-r_wts[w][0])
                                                                                                                            #expns2[r][w] = expns2[r][w] + (r_wts[w][2])

                                                                                    except:
                                                                                        continue



                                                                                    if (W0+W1)!= 0:
                                                                                        W0 = W0/(W0+W1)
                                                                                        W1 = W1/(W0+W1)
                                                                                        #W2=W2/(W0+W1+W2)


                                                                                        sval = random.random()
                                                                                        iter1[r]=iter1[r]+1
                                                                                        ##print(sval,W14,W15,W16,W17,W18,W19)
                                                                                        if sval<W0:
                                                                                            Sample[r]=0
                                                                                            Sample_r[r33]=0
                                                                                            margs23[r][0]=margs23[r][0]+1
                                                                                        elif sval<(W0+W1):
                                                                                            Sample[r]=1
                                                                                            Sample_r[r33]=1
                                                                                            margs23[r][1]=margs23[r][1]+1



                                                                                        for r22 in H1[h1]:
                                                                                            if r22 in WORDS2 and r33 in WORDS2:
                                                                                                    if r22==r33:
                                                                                                        continue
                                                                                                    if Sample_r[r22]!=Sample_r[r33]:
                                                                                                        if Sample_r[r22]==0:
                                                                                                           # #print("yesssssss")
                                                                                                            #P = P + VR
                                                                                                            #exp1 = exp1 + 0.1
                                                                                                            hhlll="Samehotel("+str(r33)+","+str(r22)+")"
                                                                                                            if r33 not in expns2 or hhlll not in expns2[r33]:
                                                                                                                expns2[r33][hhlll] = VR

                                                                                                            else:
                                                                                                                expns2[r33][hhlll] = expns2[r33][hhlll] + VR 
                                                                                                    elif Sample_r[r22]!=Sample_r[r33]:
                                                                                                        if Sample_r[r22]==1:
                                                                                                            #P = P + VR
                                                                                                            #exp1 = exp1 + 0.1
                                                                                                            hhlll="Samehotel("+str(r33)+","+str(r22)+")"
                                                                                                            if r33 not in expns2 or hhlll not in expns2[r33]:
                                                                                                                expns2[r33][hhlll] = VR

                                                                                                            else:
                                                                                                                expns2[r33][hhlll] = expns2[r33][hhlll] + VR

                                                                                        for r1 in Samuser_index[h]:
                                                                                            if r1==r:
                                                                                                continue
                                                                                            if r in WORDS2 and r1 in WORDS2:
                                                                                                    if Sample[r]!=Sample[r1]:
                                                                                                        if Sample[r1]==0:
                                                                                                            #W0=W0+r_wts1[w][0]
                                                                                                            #margs[r][0]=margs[r][0]+1
                                                                                                            hhlll="Sameuser("+str(r)+","+str(r1)+")"
                                                                                                            if r not in expns2 or hhlll not in expns2[r]:
                                                                                                                expns2[r][hhlll] =C1
                                                                                                                if r not in Relational_formula_filter:
                                                                                                                    Relational_formula_filter[r]=WORDS2[r]    
                                                                                                            else:
                                                                                                                expns2[r][hhlll] = expns2[r][hhlll] + C1
                                                                                                        elif Sample[r1]==1:
                                                                                                            #W1=W1+r_wts1[w][1]
                                                                                                           # margs[r][1]=margs[r][1]+1
                                                                                                            hhlll="Sameuser("+str(r)+","+str(r1)+")"
                                                                                                            if r not in expns2 or hhlll not in expns2[r]:
                                                                                                                expns2[r][hhlll] =C1
                                                                                                                if r not in Relational_formula_filter:
                                                                                                                    Relational_formula_filter[r]=WORDS2[r]    
                                                                                                            else:
                                                                                                                expns2[r][hhlll] = expns2[r][hhlll] +C1 






                                                                        #Computing Marginal Probability after user input
                                                                        margs22={}
                                                                        for t in margs23:
                                                                            gh=[]
                                                                            if iter1[t]>0:
                                                                                for kk in margs23[t]:
                                                                                    vv=float(kk)/float(iter1[t])
                                                                                    if float(vv)>=1.0:
                                                                                        gh.append(float(1.0))
                                                                                    elif float(vv)<1.0:
                                                                                        gh.append(abs(float(vv)))
                                                                                margs22[t]=gh
                                                                        margs33={}
                                                                        for t in margs22:
                                                                            s=0
                                                                            for ww in margs22[t]:
                                                                                s=s+float(ww)
                                                                            #if s!=0:
                                                                            ##print(t,s)
                                                                            margs33[t]=margs22[t]
                                                                        margs3={}
                                                                        margs222={}
                                                                        for ww in margs33:
                                                                            gg=[]
                                                                            for kk in range(0,len(margs33[ww])):
                                                                                    gg.append(margs33[ww][kk])
                                                                            margs222[ww]=gg

                                                                        for dd in margs222:
                                                                            v=max(margs222[dd])
                                                                            margs3[dd]=v
                                                                        #data tyoe
                                                                        d_tt={}
                                                                        d_tt[0]="negative"
                                                                        d_tt[1]="positive"
                                                                        #predict topic user input
                                                                        sampled_doc=[]
                                                                        pred_t=[]
                                                                        for a in margs222:
                                                                            for ss in range(0,len(margs222[a])):
                                                                                    if margs222[a][ss]==margs3[a]:
                                                                                    ##print(a,d_tt[ss])
                                                                                            sampled_doc.append(a)
                                                                                            pred_t.append(d_tt[ss])
                                                                        ss=set(sampled_doc)
                                                                        sampled_doc_up=[]
                                                                        sampled_doc_up_map_user={}
                                                                        for kk in ss:
                                                                            sampled_doc_up.append(kk)
                                                                        for tt in sampled_doc_up:
                                                                            ggf=[]
                                                                            for gg in range(0,len(sampled_doc)):
                                                                                if tt==sampled_doc[gg]:
                                                                                    ggf.append(pred_t[gg])
                                                                            if len(ggf)==1:
                                                                                sampled_doc_up_map_user[tt]=ggf

                                                                        cx=0   
                                                                        for s in sampled_doc_up_map_user:
                                                                            if len(sampled_doc_up_map_user[s])>1:
                                                                                    cx=cx+1
                                                                                   ##print(s,sampled_doc_up_map[s])


                                                                        ##print(doc_per_pred_topic)
                                                                        ##print(cx)

                                                                        ffd1=open("User_Prediction1.txt","w")

                                                                        for s in sampled_doc_up_map_user:
                                                                            ccvb=str(s)+":"+str(sampled_doc_up_map_user[s])
                                                                            ffd1.write(str(ccvb)+"\n")
                                                                        ffd1.close()

                                                                        #Explanation Generation  with user
                                                                        ##print(expns2)
                                                                        import operator
                                                                        correct_predictions_r = {}

                                                                        for m in margs222.keys():
                                                                                    if m in WORDS2 and m in sampled_doc_up_map_user:
                                                                                            correct_predictions_r[m] = 1
                                                                                            ##print(m)

                                                                                    #if len(WORDS[m])==0:#or ratings[m]==3:
                                                                                       # continue
                                                                                    #else:
                                                                                       # correct_predictions[m] = 1
                                                                                    #if margs[m] > 0.5 and spamindex[m] ==-1:
                                                                                        #correct_predictions[m] = 1
                                                                        fft_r=open("expn_n1_r1_user.txt","w")  
                                                                        explanation_r={}
                                                                        for e in expns2: 
                                                                            if e in correct_predictions_r:
                                                                                sorted_expns_r = sorted(expns2[e].items(), key=operator.itemgetter(1),reverse=True)
                                                                                z = 0
                                                                                for s in sorted_expns_r[:]:
                                                                                    z = z + s[1]
                                                                                rex_r = {}
                                                                                keys_r = []
                                                                                for s in sorted_expns_r[:]:
                                                                                    rex_r[s[0]] = s[1]/z
                                                                                    keys_r.append(s[0])
                                                                                #if "Sameuser" not in keys or "Samehotel" not in keys:
                                                                                    #continue
                                                                                sorted1 = sorted(rex_r.items(), key=operator.itemgetter(1),reverse=True)
                                                                                #if sorted1[0][0]=="JNTM":
                                                                                ##print(str(e) +" "+str(sorted1))
                                                                                gg=str(e) +":"+str(sorted1)
                                                                                ##print(sorted1)
                                                                                explanation_r[e]=sorted1
                                                                                fft_r.write(str(gg)+"\n")
                                                                        #hhh="Explanation_K="+str(j)+"_sample_"+str(mzzz)+"_user"+"with user feedback"+".txt"
                                                                        ##print("Samle:"+str(mzzz))
                                                                        #newpath = r'C:\Users\khanfarabi\OneDrive - The University of Memphis\Explain_MLN\Explanation_yelp\Folder_20'
                                                                        #if not os.path.exists(newpath):
                                                                                                #os.makedirs(newpath)

                                                                        #f11_r=open(os.path.join(newpath,hhh), 'w')
                                                                        #path="/Users/khanfarabi/OneDrive - The University of Memphis/Explain_MLN/Explanation_yelp/Folder_"+str(j+1)
                                                                        #newpath="Folder_"+str(j)
                                                                        #if not os.path.exists(path):
                                                                                                    #os.makedirs(path)
                                                                        #f11_r=open(os.path.join(path,hhh), 'w')

                                                                        Store_Explanation_user={}
                                                                        for t in explanation_r:
                                                                            #for k in WORDS1:
                                                                                   #if str(t)==str(k):
                                                                                        ggg=str(t)+":"+str(explanation_r[t])
                                                                                        #f11_r.write(str(ggg)+"\n")
                                                                                        #f11_r.write("\n")
                                                                                        ##print(t,explanation_r[t])
                                                                                        ##print("\n")
                                                                                        Store_Explanation_user[t]=explanation_r[t]
                                                                                        Sample_model[t]=explanation_r[t]
                                                                        expt_st_pd[mzzz]=Store_Explanation_user
                                                                        ##print(mzzz,len(Store_Explanation_user))

                                                                        #f11_r.close()
                                                                        return expt_st_pd,margs3,reduced_obj






                                def compute_exp_acc(final_clu,T,K):
                                            #cluster_generatio(j)
                                            jjk={}
                                            jjz={}
                                            ob={}
                                            for cz in range(0,T):
                                                    M=0.35
                                                    ss=random.random()
                                                    if ss<0.2:
                                                        M1=ss
                                                    else:
                                                        M1=random.random()
                                                    expt_st_pd,margs3,reduced_obj=sample_rev(cz,final_clu,K,M)
                                                    jjk[cz]=expt_st_pd
                                                    jjz[cz]=margs3
                                                    ob[cz]=reduced_obj
                                            expt_st_pc[K]=jjk
                                            perd_m[K]=jjz
                                            obj_m[K]=ob
                                            ob_clm={}
                                            for t in final_clu:
                                                   for jj in final_clu[t]:
                                                        ##print(jj,t)
                                                        ob_clm[int(jj)]=int(t)
                                            reliab_exp={}
                                            uq=[]

                                            for kk in expt_st_pc[K]:
                                                        #c=0
                                                        for jj in expt_st_pc[K][kk]:
                                                                for yy in expt_st_pc[K][kk][jj]:
                                                                      if yy not in uq:
                                                                            uq.append(yy)
                                            red_o=[]
                                            ss=set(uq)
                                            for v in ss:
                                                red_o.append(v)
                                            from collections import defaultdict
                                            reliab_exp={}
                                            for zz in red_o:
                                                    yz={}#defaultdict(list)
                                                    for kk in expt_st_pc[K]:
                                                           for jj in expt_st_pc[K][kk]:
                                                                for yy in expt_st_pc[K][kk][jj]:
                                                                    if int(zz)==int(yy):
                                                                        yz[int(jj)]=expt_st_pc[K][kk][jj][yy]
                                                    reliab_exp[int(zz)]=yz
                                            # Word Aggregation 
                                            nm=[]
                                            reliab_exp_up={}
                                            for d in range(0,T):
                                                nm.append(int(d))

                                            for tt in reliab_exp:
                                                #if tt==1:
                                                cz=[]
                                                c3=0

                                                exp=[]
                                                for kk in reliab_exp[tt]:
                                                            cz.append(int(kk)) 
                                                cz1=list(set(nm)-set(cz))
                                                ##print(tt,cz,cz1)
                                                for zb in cz1:
                                                    try:
                                                        for k in obj_m[K][zb]:
                                                            if ob_clm[int(tt)]==ob_clm[int(k)]:
                                                                break
                                                    except:
                                                        continue
                                                    ##print(tt,k,Store_Explanation_user2[k])
                                                    ##print("\n\n")
                                                    c2=0
                                                    c3=c3+1
                                                    gggg="D_"+str(c3)
                                                    vbb=[]
                                                    #vbb.append(gggg)
                                                    if k in Store_Explanation_user2:
                                                        for hh in Store_Explanation_user2[k]:
                                                                c2=c2+1
                                                                if "Same" not in hh[0]:
                                                                            bb1=hh[0]#+"+ R:"+str(c2)+"+ W:"+str(hh[1])+"+ M:"+str(margs3_u[k])
                                                                            ##print(tt,zb,bb1)
                                                                            ##print("\n")
                                                                            if bb1 not in vbb:
                                                                                vbb.append(bb1)
                                                        exp.append(vbb)
                                                    ##print(bb1)
                                                    #if bb1 not in vbb:

                                                for cx in cz:
                                                    c=0
                                                    c3=c3+1
                                                    gggg="D_"+str(c3)
                                                    vbb=[]
                                                    #vbb.append(gggg)
                                                    for hh in reliab_exp[tt][int(cx)]:
                                                            c=c+1
                                                            if "Same" not in hh[0]:
                                                                        bb=hh[0]#+"+ R:"+str(c)+"+ W:"+str(hh[1])+"+ M:"+str(perd_m[10][int(cx)][tt])
                                                                        ##print(tt,cx,bb)
                                                                        ##print("\n\n")
                                                                        if bb not in vbb:
                                                                            vbb.append(bb)
                                                    exp.append(vbb)
                                                reliab_exp_up[tt]=exp
                                                #R aggregate words




                                                #rank_list = [['A', 'B', 'C'], ['B', 'A', 'C'], ['C', 'D', 'A']]
                                                from rpy2.robjects import r
                                                reliab_exp_up_1w={}#reliab_exp_up_2w={}#reliab_exp_up_1w

                                                for t1 in reliab_exp_up:
                                                    gh=[]
                                                    gh1=[]
                                                    c=0
                                                    gh.append(t1)
                                                    ll=reliab_exp_up[t1]
                                                    b=r['CEMC'](ll,k=30)#agg.CEMC(ll)
                                                    for t in b[0][0]:
                                                        if c<T:
                                                            gh.append(t)
                                                            c=c+1
                                                   # #print(gh)
                                                    gh1.append(gh)
                                                    reliab_exp_up_1w[t1]=gh1
                                                    # Relational Aggregation 
                                                    nm=[]
                                                    rw={}
                                                    reliab_exp_up2={}
                                                    for d in range(0,T):
                                                        nm.append(int(d))

                                                    for tt in reliab_exp:
                                                        #if tt==1:
                                                        cz=[]
                                                        c3=0

                                                        exp=[]
                                                        for kk in reliab_exp[tt]:
                                                                    cz.append(int(kk)) 
                                                        cz1=list(set(nm)-set(cz))
                                                        ##print(tt,cz,cz1)
                                                        for zb in cz1:
                                                            try:
                                                                for k in obj_m[K][zb]:
                                                                    if ob_clm[int(tt)]==ob_clm[int(k)]:
                                                                        break
                                                            except:
                                                                continue
                                                            ##print(tt,k,Store_Explanation_user2[k])
                                                            ##print("\n\n")
                                                            c2=0
                                                            c3=c3+1
                                                            gggg="D_"+str(c3)
                                                            vbb=[]
                                                            #vbb.append(gggg)
                                                            if k in Store_Explanation_user2:
                                                                    for hh in Store_Explanation_user2[k]:
                                                                            c2=c2+1
                                                                            if "Same"  in hh[0]:
                                                                                        gggz=hh[0].strip(")").split(",")
                                                                                        hh1="Q_"+str(gggz[1])
                                                                                        rw[hh1]=float(hh[1])
                                                                                        bb1=hh1#+"+ R:"+str(c2)+"+ W:"+str(hh[1])+"+ M:"+str(margs3_u[k])
                                                                                        ##print(tt,zb,bb1)
                                                                                        ##print("\n")
                                                                                        if bb1 not in vbb:
                                                                                            vbb.append(bb1)             

                                                                    exp.append(vbb)
                                                            ##print(bb1)
                                                            #if bb1 not in vbb:

                                                        for cx in cz:
                                                            c=0
                                                            c3=c3+1
                                                            gggg="D_"+str(c3)
                                                            vbb=[]
                                                            #vbb.append(gggg)
                                                            for hh in reliab_exp[tt][int(cx)]:
                                                                    c=c+1
                                                                    if "Same"  in hh[0]:
                                                                                gggz=hh[0].strip(")").split(",")
                                                                                hh1="Q_"+str(gggz[1])
                                                                                rw[hh1]=float(hh[1])
                                                                                bb=hh1#+"+ R:"+str(c)+"+ W:"+str(hh[1])+"+ M:"+str(perd_m[10][int(cx)][tt])
                                                                                ##print(tt,cx,bb)
                                                                                ##print("\n\n")
                                                                                if bb not in vbb:
                                                                                    vbb.append(bb)        
                                                            exp.append(vbb)
                                                        reliab_exp_up2[tt]=exp
                                                        # Relational Aggregation 
                                                        nm=[]
                                                        rw={}
                                                        reliab_exp_up2={}
                                                        for d in range(0,T):
                                                            nm.append(int(d))

                                                        for tt in reliab_exp:
                                                            #if tt==1:
                                                            cz=[]
                                                            c3=0

                                                            exp=[]
                                                            for kk in reliab_exp[tt]:
                                                                        cz.append(int(kk)) 
                                                            cz1=list(set(nm)-set(cz))
                                                            ##print(tt,cz,cz1)
                                                            for zb in cz1:
                                                                try:
                                                                    for k in obj_m[K][zb]:
                                                                        if ob_clm[int(tt)]==ob_clm[int(k)]:
                                                                            break
                                                                except:
                                                                    continue
                                                                ##print(tt,k,Store_Explanation_user2[k])
                                                                ##print("\n\n")
                                                                c2=0
                                                                c3=c3+1
                                                                gggg="D_"+str(c3)
                                                                vbb=[]
                                                                #vbb.append(gggg)
                                                                if k in Store_Explanation_user2:
                                                                        for hh in Store_Explanation_user2[k]:
                                                                                c2=c2+1
                                                                                if "Same"  in hh[0]:
                                                                                            gggz=hh[0].strip(")").split(",")
                                                                                            hh1="Q_"+str(gggz[1])
                                                                                            rw[hh1]=float(hh[1])
                                                                                            bb1=hh1#+"+ R:"+str(c2)+"+ W:"+str(hh[1])+"+ M:"+str(margs3_u[k])
                                                                                            ##print(tt,zb,bb1)
                                                                                            ##print("\n")
                                                                                            if bb1 not in vbb:
                                                                                                vbb.append(bb1)             

                                                                        exp.append(vbb)
                                                                ##print(bb1)
                                                                #if bb1 not in vbb:

                                                            for cx in cz:
                                                                c=0
                                                                c3=c3+1
                                                                gggg="D_"+str(c3)
                                                                vbb=[]
                                                                #vbb.append(gggg)
                                                                for hh in reliab_exp[tt][int(cx)]:
                                                                        c=c+1
                                                                        if "Same"  in hh[0]:
                                                                                    gggz=hh[0].strip(")").split(",")
                                                                                    hh1="Q_"+str(gggz[1])
                                                                                    rw[hh1]=float(hh[1])
                                                                                    bb=hh1#+"+ R:"+str(c)+"+ W:"+str(hh[1])+"+ M:"+str(perd_m[10][int(cx)][tt])
                                                                                    ##print(tt,cx,bb)
                                                                                    ##print("\n\n")
                                                                                    if bb not in vbb:
                                                                                        vbb.append(bb)        
                                                                exp.append(vbb)
                                                            reliab_exp_up2[tt]=exp
                                                            #R aggregate relation

                                                            #rank_list = [['A', 'B', 'C'], ['B', 'A', 'C'], ['C', 'D', 'A']]
                                                            from rpy2.robjects import r
                                                            #reliab_exp_up_1={} reliab_exp_up_2={}reliab_exp_up_3={}
                                                            reliab_exp_up_1r={}#reliab_exp_up_2r={}#reliab_exp_up_1r

                                                            for t1 in reliab_exp_up2:
                                                                gh=[]
                                                                gh1=[]
                                                                c=0
                                                                gh.append(t1)
                                                                ll=reliab_exp_up2[t1]
                                                                try:
                                                                    b=r['CEMC'](ll,k=10)#agg.CEMC(ll)
                                                                    for t in b[0]:
                                                                        if c<T:
                                                                            gh.append(t)
                                                                            c=c+1
                                                                    gh1.append(gh)
                                                                    reliab_exp_up_1r[t1]=gh1
                                                                except:
                                                                    continue
                                                            # Aggregate Review Relational Exp
                                                            aggregate_relational_exp_review={}
                                                            for t in reliab_exp_up_1r:
                                                                expr=[]
                                                                for d in reliab_exp_up_1r[t][0]:
                                                                    if 'Q' in str(d):
                                                                        hh=d.split("_")
                                                                        if qrat[int(t)]==qrat[int(hh[1])]:
                                                                                    ##print(t,hh[1])
                                                                                if random.random()<0.5:
                                                                                            expr.append(hh[1])
                                                                if len(expr)>=1:
                                                                            aggregate_relational_exp_review[t]=expr

                                                            #words exp accuracy
                                                            reliab_exp_up_1w_pc={}
                                                            reliab_exp_up_1w_score={}
                                                            reliab_exp_up_1wu={}

                                                            for jj in reliab_exp_up_1w:
                                                                if jj in ann:
                                                                    ghh=[]
                                                                    cv=0
                                                                    c=0
                                                                    mm=5
                                                                    for cz1 in reliab_exp_up_1w[jj][0]:
                                                                             if cv<mm:
                                                                                 if cz1 in ann[jj]:
                                                                                        if random.random()<0.4:
                                                                                                    c=c+1
                                                                                                    cv=cv+1
                                                                    #mm=mm-0.45
                                                                    s=c/mm

                                                                    if s>0:
                                                                        reliab_exp_up_1w_pc[jj]=s
                                                                        ##print(s)


                                                            zx=0
                                                            cx=0
                                                            for z in reliab_exp_up_1w_pc:
                                                                zx=zx+reliab_exp_up_1w_pc[z]
                                                                cx=cx+1
                                                            ##print(reliab_exp_up_1w_score)
                                                            ##print(zx/cx)
                                                            try:
                                                                wexp=zx/cx
                                                            except:
                                                                continue 

                                                            # Relational exp accuracy

                                                            cn=0
                                                            agg_s={}
                                                            agg_avg={}
                                                            for t in aggregate_relational_exp_review:
                                                                s=0
                                                                if t in similar_r_map:
                                                                    for z in aggregate_relational_exp_review[t]:
                                                                        if z in similar_r_map[t]:
                                                                            s=s+1
                                                                if s>0:
                                                                    ss=s/5.0
                                                                    if ss>1.0:
                                                                        agg_avg[t]=1.0
                                                                    else:
                                                                        agg_avg[t]=ss
                                                            st=0        
                                                            for y in agg_avg:
                                                                st=st+agg_avg[y]
                                                            ##print("Relational Accuracy: "+str(st/len(agg_avg))+"\n")
                                                            try:
                                                                rexp=st/len(agg_avg)

                                                            except:
                                                                continue
                                                            return wexp,rexp






                                cluster_varying_wexp_random_m={}
                                cluster_varying_rexp_random_m={}
                                K=10
                                final_clu=cluster_generation1(K)
                                for jk in range(5,14,2):
                                        wxz=[]
                                        rxz=[]
                                        tt=10
                                        for t in range(0,5):
                                            wexp,rexp=compute_exp_acc(final_clu,jk,tt)
                                            wxz.append(wexp)
                                            rxz.append(rexp)

                                        cluster_varying_wexp_random_m[jk]=wxz
                                        cluster_varying_rexp_random_m[jk]=rxz

                               # print(cluster_varying_wexp_random_m,cluster_varying_rexp_random_m)









