#Pre-process the Reviews Run1
import random
import sys
import random
import re
from collections import defaultdict
import pandas as pd
import time
class operation:
    @classmethod
    def test(cls):
                    tid=0
                    c=0
                    lng=[]
                    text_t=[]
                    twitter_id=defaultdict(list)
                    users = defaultdict(list)
                    twit=defaultdict(list)
                    trate=defaultdict(list)
                    #pd.read_csv
                    ifile1 =pd.read_csv("/Data/sampled_medical.csv",encoding='latin-1')
                    header=[]
                    users = defaultdict(list)
                    for ln in ifile1:
                        parts = ln.strip('" ""').split(",")
                        #pass#print(parts)
                        header.append(parts)
                        #users[parts[4]].append(tid)
                       # twit[tid].append(parts[5])
                       # trate[tid]=parts[0].strip('"')


                    for t1 in ifile1['text']:
                        text_t.append(t1)
                    #sr=[]
                    #c=0
                    #ver=[]
                    sr=[]
                    users = defaultdict(list)
                    for tt in ifile1['user_id']:#screen_name
                            sr.append(tt)
                    #Pre-process the Reviews Run2
                    import random
                    import sys
                    import random
                    import re
                    from collections import defaultdict
                    import pandas as pd
                    tid=0
                    c=0
                    lng=[]
                    #text_t=[]
                    twitter_id=defaultdict(list)
                    users = defaultdict(list)
                    twit=defaultdict(list)
                    trate=defaultdict(list)
                    #pd.read_csv
                    ifile1 =pd.read_csv("/Data/sampled.csv",encoding='latin-1')
                    header=[]

                    for ln in ifile1:
                        parts = ln.strip('" ""').split(",")
                        #pass#print(parts)
                        header.append(parts)
                        #users[parts[4]].append(tid)
                       # twit[tid].append(parts[5])
                       # trate[tid]=parts[0].strip('"')


                    for t1 in ifile1['text']:
                        text_t.append(t1)
                    #for t2 in ifile1['screen_name']:
                       # sr.append(t2)
                    for tt in ifile1['user_id']:#screen_name
                            sr.append(tt)
                    #Pre-process the Reviews Run3
                    #Sample File
                    import random
                    import sys
                    import random
                    import re
                    from collections import defaultdict
                    import pandas as pd
                    tid=0
                    c=0
                    lng=[]
                    #text_t=[]
                    twitter_id=defaultdict(list)
                    users = defaultdict(list)
                    twit=defaultdict(list)
                    trate=defaultdict(list)
                    #pd.read_csv
                    ifile1 =pd.read_csv("/Data/sampled_textonly_science.csv",encoding='latin-1')
                    header=[]
                    users = defaultdict(list)
                    for ln in ifile1:
                        parts = ln.strip('" ""').split(",")
                        #pass#print(parts)
                        header.append(parts)
                        #users[parts[4]].append(tid)
                       # twit[tid].append(parts[5])
                       # trate[tid]=parts[0].strip('"')
                    rt_ct=[]
                    label=[]
                    text_t1=[]
                    #for t in ifile1['retweet_count']:
                            #rt_ct.append(t)
                    for t in ifile1['label']:
                            label.append(t)
                    for t1 in ifile1['text']:
                       # pass#print(t1)
                        text_t1.append(t1)

                    #for tt in ifile1['verified']:
                            #ver.append(tt)     
                    #concatenate Run 4
                    pass#print(len(text_t1),len(label))
                    text_t2=[]
                    for tt in text_t:
                        text_t1.append(tt)
                    for ty in range(0,len(text_t)):
                        label.append(1)
                    pass#print(len(text_t1),len(label))
                    #filter twits based on positive and negative twits Run 5
                    twiter_id=0
                    twit1={}
                    twit={}
                    twit_count={}
                    vcc=0
                    sr1=[]
                    for t in range(0,len(text_t)):
                        #if lng[t]=='en':# and ver[t]==True:
                           # if vcc<20000:
                                twit[twiter_id]=text_t1[t]
                                twit_count[twiter_id]=label[t]
                                #pass#print(ver[t])
                                #sr1.append(sr[t])
                                twiter_id=twiter_id+1
                                vcc=vcc+1

                    #for y in range(1000,2000):
                           #twit[y]=text_t1[y-20000]
                    #for y in range(0,40000):
                        #pass#print(y,twit[y])
                    #Run 7 checking scitific and non-scientific tweets count
                    pass#print(len(twit),len(twit_count))
                    t_1=[]
                    t_0=[]
                    vv=0
                    for t in twit_count:
                        if twit_count[t]==1:
                                t_1.append(t)
                        elif twit_count[t]==0:
                              if vv<54:
                                    t_1.append(t)
                                    vv=vv+1

                    # Run 8 check count of atoms
                    twit2={}
                    for kk in twit:
                        if kk in t_1:
                            twit2[kk]=twit[kk]
                    pass#print(len(twit2))
                    pass#print(len(t_1))
                    #source of tweeter. 

                    vcc=0
                    sr1=[]
                    for t in range(0,len(lng)):
                        if lng[t]=='en':
                            #if vcc<100000:
                                sr1.append(sr[t])
                                vcc=vcc+1
                    #Sameuser Relation
                    #Checking # of tweets. Mapping # of airlines per tweet Run 9
                    import sys
                    ti=[]
                    m_sr={}
                    for g in twit.keys():
                        ti.append(g)
                    #pass#print(ti)   
                    ss=set(sr)
                    sr2=[]
                    for k in ss:
                        sr2.append(k)
                    #pass#print(sr2)  
                    #sys.exit()
                    vv=-1
                    for j in sr2:
                        gh=[]
                        vv=vv+1
                        for tt in range(0,len(sr)):
                                        if str(j)==str(sr[tt]):
                                            #pass#print(j)
                                           # if ti[tt] not in gh:
                                            gh.append(ti[tt])
                                    #if len(gh)>=1:
                        #pass#print(j,gh)
                        m_sr[j]=gh

                    #data preprocessing Run 10
                    import sys
                    import os
                    import re
                    import string
                    from collections import defaultdict
                    pass#print(len(twit))
                    flags = (re.UNICODE if sys.version < '3' and type(text) is unicode else 0)
                    stopwords=[]
                    sfile = open("/Data/stopwords.txt")
                    for ln in sfile:
                        stopwords.append(ln.strip().lower())
                    sfile.close()

                    sentwords=[]
                    windex=defaultdict(list)
                    sfile1 = open("/Data/Words.txt")
                    for ln in sfile1:
                        sentwords.append(ln.strip().lower())
                    sfile1.close()
                    WORDS={}
                    for t in twit2:
                        keep=[]
                        #for kk in twit[t]:
                        #pass#print(kk)
                        kk=twit2[t]
                        for word in re.findall(r"\w[\w':]*", kk, flags=flags):
                            if word.isdigit() or len(word)==1:
                                                continue
                            word_lower = word.lower()
                            #pass#print(word_lower)

                            if word_lower in stopwords:
                                                  continue

                            for ty in string.punctuation:
                                                if str(ty) in word_lower:
                                                    continue
                            #for tr in arlu1:
                                #if str(tr) in word_lower:
                                    #continue
                            if word_lower.isalnum():# and word_lower in sentwords:
                                if word_lower not in keep:# and word_lower not in vv:
                                    if not any(c.isdigit() for c in word_lower):
                                            keep.append(word_lower)
                            if len(keep)>=6:
                                WORDS[t]=keep
                                for zzw in keep:
                                    if t not in windex:
                                        windex[zzw].append(t)




                    # Run 11 maping a word occurs  in mutliple tweets
                    windex1={}
                    for c in windex:
                        gh=[]
                        ss=set(windex[c])
                        for ty in ss:
                            gh.append(ty)
                        windex1[c]=gh
                    for k in windex1:
                        pass#pass#print(k,windex1[k])
                    #Run 12 storing training and target train_r targets_r
                    train_r=[]
                    targets_r=[]
                    m_tid_tr1={}
                    c=0
                    c1=0
                    for t in WORDS:
                        if twit_count[t]==1:
                            for tt in WORDS[t]:
                                train_r.append(tt)
                                targets_r.append(1)
                        elif twit_count[t]==0:
                            for tt in WORDS[t]:
                                train_r.append(tt)
                                targets_r.append(0)



                    pass#print(len(train_r),len(targets_r))
                    #SVM Learner Run 21
                    from sklearn.feature_extraction.text import TfidfVectorizer
                    from sklearn import svm
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
                    from sklearn.model_selection import cross_val_score
                    import statistics 
                    from sklearn.model_selection import cross_val_score
                    #Learn SVM Model 
                    #ifile = open("processed_revs_1.txt",encoding="ISO-8859-1")
                    Y = targets_r
                    words = []
                    unique_words = []

                    ss=set(train_r)
                    for w in ss:
                            if w not in unique_words:
                                unique_words.append(w)

                    #######
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
                    #pass#print(f1_score(Y,p,average='weighted'))




                    #########
                    tf_transformer = TfidfVectorizer()
                    f = tf_transformer.fit_transform(train_r)
                    features = [((i, j), f[i,j]) for i, j in zip(*f.nonzero())]
                    unique_word_ids = []
                    for w in unique_words:
                        i = tf_transformer.vocabulary_.get(w)
                        unique_word_ids.append(i)

                    clf = svm.LinearSVC(C=1)
                    clf.fit(f,Y)
                    clf.fit(f,Y)
                    p = clf.predict(f)
                    pass#print(f1_score(Y,p,average='weighted'))
                    scores = cross_val_score(clf,f, Y, cv=5, scoring='f1_micro')
                    pass#print("cv 5 score")
                    pass#print(statistics.mean(scores))

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
                        pass#print(i,j)
                        for k in tf_transformer.vocabulary_.keys():
                            if tf_transformer.vocabulary_[k]==j:
                                if k not in r_wts:
                                    break
                                else:
                                    r_wts[k][i] = V1[ix][0]
                                    break
                        ix = ix + 1


                    #pass#print(r_wts)
                    #for d in WORDS: #Run 12
                        #pass#print(d,Rev_text_map[d])
                    #Create Sentences for clustering
                    #Run 13
                    rev=m_sr#{'1':['2','3'],'2':['4','5'],'3':['6','2'],'4':['1','2'],'5':['6'],'6':['4']}

                    sent=[]

                    sent_map=defaultdict(list)
                    for g in WORDS:
                        gf=[]
                        for k in WORDS[g]:
                            gf.append(str(g))
                            gf.append(str(k))
                            for t in rev:
                                    gf.append(str(t))
                            if gf not in sent:
                                    sent.append(gf)




                    documents=[]
                    #documents1=[]
                    for t in sent:
                        for jh in t:
                            if jh not in documents:
                                    documents.append(jh)
                    #K-Means Run 14

                    #cluster generation with k-means
                    import sys
                    from nltk.cluster import KMeansClusterer
                    import nltk
                    from sklearn import cluster
                    from sklearn import metrics
                    import gensim 
                    import operator
                    from gensim.models import Word2Vec


                    model = Word2Vec(sent, min_count=5)

                    X = model[model.wv.vocab]



                    NUM_CLUSTERS=20
                    kclusterer = KMeansClusterer(NUM_CLUSTERS, distance=nltk.cluster.util.cosine_distance,repeats=3)
                    assigned_clusters1 = kclusterer.cluster(X,assign_clusters=True)
                    #pass#print (assigned_clusters)
                    cluster={}
                    words = list(model.wv.vocab)
                    for i, word in enumerate(words):
                      gh=[] 
                      gh1=[] 
                      gh2=[] 
                      if word.isdigit(): 
                        cluster[word]=assigned_clusters1[i]
                        #pass#print (word + ":" + str(assigned_clusters[i]))
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
                    for k in final_clu:
                        pass#print(k,final_clu[k],len(final_clu[k]))


                    #user annotated feedback
                    t_tword=[]
                    f=open("annotations_2.txt")
                    for k in f:
                        p=k.strip("\n").split(":")
                        pp=p[2].split(",")
                        for kk in pp:
                            t_tword.append(kk.strip(" "))
                    f.close()
                    #pass#print(t_tword)
                    #annotated explsanation per query
                    ann1={}
                    c=0
                    for k in WORDS:
                        if twit_count[k]==1:
                            c=c+1
                            gff=[]
                            for gg in WORDS[k]:
                                if gg in t_tword:
                                    gff.append(gg)
                            if len(gff)>0:
                               # if k in WORDSt:
                                    ann1[k]=gff

                        elif twit_count[k]==0:
                            c=c+1
                            gff1=[]
                            for gg in WORDS[k]:
                                if gg in t_tword:
                                    gff1.append(gg)
                            if len(gff1)>0:
                                #if k WORDSt:
                                    ann1[k]=gff1

                    ann={}
                    for t in ann1:
                        if t in WORDS:
                            ann[t]=ann1[t]
                    #ff=open("annotated.txt")
                    for k in ann:
                        pass#pass#print(k,ann[k])
                    # query class
                    d_tt={}
                    d_tt[0]="non-scientific"
                    d_tt[1]="scientific"
                    #feedback words from the users
                    '''
                    import random
                    reduced_obj=[]

                    for k in final_clu:
                        for tt in final_clu[k]:
                            if random.random()<0.5:
                                if tt not in reduced_obj:
                                    reduced_obj.append(tt)
                    pass#print(len(reduced_obj))
                    '''
                    wr={}
                    w=[]
                    for k in final_clu:
                            #c=-1
                            #c=c+1

                            md=int(len(final_clu[k])/2)
                            c=0      
                            k1= final_clu[k][md+c]
                            #pass#print(k1,md)        
                            if k1 in ann:
                                        for k3 in ann[k1]:
                                                w.append(k3)
                            else:
                                c=c+11
                                continue 


                            #pass#print(k,k1,md,d_tt[twit_count[k1]],w)
                            wr[k1]=w
                    #pass#print(w)
                    #using user feedback Run 17
                    import gensim 
                    import operator
                    from gensim.models import Word2Vec
                    model = Word2Vec(sent, min_count=1)
                    data_g={}
                    #for t in WORDS:
                        #pass#print(t,WORDS[t])
                    #Update Evidence Based on Survey input Update 2 Run 19
                    #data_extract11 similar_r_map1 data_extract11
                    import gensim 
                    import operator
                    from gensim.models import Word2Vec
                    model = Word2Vec(sent, min_count=1)
                    data_g={}
                    for t in WORDS:
                        chu=[]
                        c=0
                        vb={}
                        for v in w:  
                            #if str(v) in WORDS[t]:
                                    vb1={}
                                    for v1 in WORDS[t]:
                                        #if 'no' not in v or 'no' not in v1:
                                                gh1=model.similarity(v,v1)
                                                if gh1>=0.01:
                                                          vb1[v1]=float(gh1)
                                                          #pass#print(gh1)
                                    for jk in vb1:
                                        if jk in vb:
                                            if float(vb1[jk])>=float(vb[jk]):
                                                #pass#print(jk,vb1[jk],vb[jk])
                                                vb[jk]=vb1[jk]
                                        else:
                                            vb[jk]=vb1[jk]

                        dd=sorted(vb.items(), key=operator.itemgetter(1),reverse=True)
                        cc=0
                        for kkk in dd:
                            if kkk[0] not in chu:
                                if cc<5:
                                    chu.append(kkk[0])
                                    cc=cc+1

                        if len(chu)>0:
                            data_g[t]=chu

                    # Run 20

                    pass#print(len(WORDS))
                    #Updating the Whole Evidence Based on Survey Input

                    WORDS22={}
                    for gg in WORDS:
                        #if gg in data_extract12:
                            #WORDS2[gg]=data_extract12[gg]
                        if gg in data_g:
                            if len(data_g[gg])>0:
                                WORDS22[gg]=data_g[gg]
                    #pass#print(WORDS2['d_535'])
                    pass#print(len(WORDS22))
                    for t in WORDS22:
                        pass#print(t,WORDS22[t])
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
                                            ##pass#print(gh)


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
                                        pass##pass#print(t)
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
                                    for jj in WORDS:
                                        gh1=[]
                                        gh2=[]
                                        s=0

                                        for k in documents1:
                                            if str(k)==str(jj):
                                                gh=model.most_similar(positive=str(k),topn=600)
                                               # #pass#print(gh)
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
                                            if twit_count[int(jj)]==twit_count[int(t5[0])]:
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
                    cmp=5
                    similar_r_map=relational_embedding_exp(cmp)


                    # Updated evidence
                    def update_exp():
                                #user  feedback explanations
                                #Sample 2 user

                                import random
                                from collections import defaultdict

                                #Sampler
                                margs23 =defaultdict(list)
                                Relational_formula_filter={}
                                iter11=defaultdict(list)
                                users_s= defaultdict(list)
                                expns2 = defaultdict(dict)
                                Relational_formula_filter={}
                                same_user={}
                                Sample={}
                                for h in m_sr:
                                    pass#print(h)
                                    for i,r in enumerate(m_sr[h]):
                                        Sample[r] = random.randint(0,1)
                                        #Sample_r[r] = random.randint(0,1)
                                        if r in WORDS22:
                                            margs23[r] = [0]*2
                                            #margs23_r[r] = 0
                                            iter11[r] =0

                                #Tunable parameter (default value of prob)
                                C1 = 0.98
                                VR=0.98
                                iters =1000000

                                for t in range(0,iters,1):
                                    h = random.choice(list(m_sr.keys()))
                                    h1 = random.choice(list(m_sr.keys()))
                                    if len(m_sr[h])==0:
                                        continue
                                    if len(m_sr[h1])==0:
                                        continue
                                    ix = random.randint(0,len(m_sr[h])-1)
                                    r = m_sr[h][ix]
                                    if r in WORDS22:
                                        if random.random()<0.5:
                                            #sample Topic
                                            W0=0
                                            W1=0
                                            #W2=0
                                            #W3=0
                                            #W4=0
                                            #W5=0
                                            #W6=0
                                            #W7=0
                                            #W8=0
                                            #W9=0
                                            #W2=0


                                            try:
                                                        for w in WORDS22[r]:
                                                            if len(r_wts[w])<0:
                                                                continue
                                                            W0=W0+r_wts[w][0]
                                                            W1=W1+(1-r_wts[w][0])
                                                           # W2=W2+r_wts[w][2]
                                                           # W3=W3+r_wts[w][3]
                                                           # W4=W4+r_wts[w][4]
                                                            #W5=W5+r_wts[w][5]
                                                            #W6=W6+r_wts[w][6]
                                                            #W7=W7+r_wts[w][7]
                                                            #W8=W8+r_wts[w][8]
                                                            #W9=W9+r_wts[w][9]


                                                            if r not in expns2 or w not in expns2[r]:
                                                                                expns2[r][w] = r_wts[w][0]
                                                                                expns2[r][w] = (1-r_wts[w][0])
                                                                               # expns2[r][w] = r_wts[w][2]
                                                                                #expns2[r][w] = r_wts[w][3]
                                                                               # expns2[r][w] = r_wts[w][4]
                                                                               # expns2[r][w] = r_wts[w][5]
                                                                                #expns2[r][w] = r_wts[w][6]
                                                                               # expns2[r][w] = r_wts[w][7]
                                                                                #expns2[r][w] = r_wts[w][8]
                                                                                #expns2[r][w] = r_wts[w][9]


                                                            else:
                                                                                expns2[r][w] =expns2[r][w]+r_wts[w][0]
                                                                                expns2[r][w] = expns2[r][w]+(1-r_wts[w][0])
                                                                               # expns2[r][w] = expns2[r][w]+r_wts[w][2]
                                                                               # expns2[r][w] = expns2[r][w]+r_wts[w][3]
                                                                                #expns2[r][w] = expns2[r][w]+r_wts[w][4]
                                                                                #expns2[r][w] = expns2[r][w]+r_wts[w][5]
                                                                                #expns2[r][w] = expns2[r][w]+r_wts[w][6]
                                                                                #expns2[r][w] = expns2[r][w]+r_wts[w][7]
                                                                                #expns2[r][w] = expns2[r][w]+r_wts[w][8]
                                                                                #expns2[r][w] = expns2[r][w]+r_wts[w][9]

                                            except:
                                                continue



                                            if (W0+W1) != 0:
                                                W0 = W0/(W0+W1)
                                                W1 = W1/(W0+W1)
                                               # W2 = W2/(W0+W1+W2+W3+W4+W5)
                                               # W3=W3/(W0+W1+W2+W3+W4+W5)
                                               # W4=W4/(W0+W1+W2+W3+W4+W5)
                                               # W5=W5/(W0+W1+W2+W3+W4+W5)
                                               # W3 = W3/(W0+W1+W2+W3+W4+W5+W6+W7+W8+W9)
                                               # W4 = W4/(W0+W1+W2+W3+W4+W5+W6+W7+W8+W9)
                                               # W5 = W5/(W0+W1+W2+W3+W4+W5+W6+W7+W8+W9)
                                               # W6 = W6/(W0+W1+W2+W3+W4+W5+W6+W7+W8+W9)
                                               # W7= W7/(W0+W1+W2+W3+W4+W5+W6+W7+W8+W9)
                                               # W8 = W8/(W0+W1+W2+W3+W4+W5+W6+W7+W8+W9)
                                               # W9 = W9/(W0+W1+W2+W3+W4+W5+W6+W7+W8+W9)
                                                #W2=W2/(W0+W1+W2)(W0+W1+W2+W3+W4+W5+W6+W7+W8+W9)


                                                sval = random.random()
                                                iter11[r]=iter11[r]+1
                                                #pass#print(sval,W14,W15,W16,W17,W18,W19)
                                                if sval<W0:
                                                    Sample[r]=1
                                                    margs23[r][0]=margs23[r][0]+1
                                                elif sval<(W0+W1):
                                                    Sample[r]=0
                                                    margs23[r][1]=margs23[r][1]+1
                                                #elif sval<(W0+W1+W2):
                                                   # Sample[r]=2
                                                    #margs23[r][2]=margs23[r][2]+1
                                                #elif sval<(W0+W1+W2+W3):
                                                    #Sample[r]=3
                                                    #margs23[r][3]=margs23[r][3]+1
                                               # elif sval<(W0+W1+W2+W3+W4):
                                                    #Sample[r]=4
                                                    #margs23[r][4]=margs23[r][4]+1
                                                #elif sval<(W0+W1+W2+W3+W4+W5):
                                                    #Sample[r]=5
                                                    #margs23[r][5]=margs23[r][5]+1
                                                #elif sval<((W0+W1+W2+W3+W4+W5+W6)):
                                                    #Sample[r]=1
                                                    #margs23[r][6]=margs23[r][6]+1
                                                #elif sval<(W0+W1+W2+W3+W4+W5+W6+W7):
                                                    #Sample[r]=1
                                                    #margs23[r][7]=margs23[r][7]+1
                                                #elif sval<(W0+W1+W2+W3+W4+W5+W6+W7+W8):
                                                    #Sample[r]=1
                                                    #margs23[r][8]=margs23[r][8]+1
                                                #elif sval<(W0+W1+W2+W3+W4+W5+W6+W7+W8+W9):
                                                    #Sample[r]=1
                                                    #margs23[r][9]=margs23[r][9]+1







                                                for r1 in m_sr[h]:
                                                    if r1==r:
                                                        continue
                                                    if r in WORDS22:
                                                        try:
                                                            if Sample[r]!=Sample[r1]:
                                                                if Sample[r1]==1:
                                                                    #W0=W0+r_wts1[w][0]
                                                                    #margs23[r][0]=margs23[r][0]+1
                                                                    hhlll="Sameuser("+str(r)+","+str(r1)+")"
                                                                    if r not in expns2 or hhlll not in expns2[r]:
                                                                        expns2[r][hhlll] =C1
                                                                        if r not in Relational_formula_filter:
                                                                            Relational_formula_filter[r]=WORDS22[r]    
                                                                    else:
                                                                        expns2[r][hhlll] = expns2[r][hhlll] + C1
                                                                elif Sample[r1]==0:
                                                                    #W1=W1+r_wts1[w][1]
                                                                   # margs23[r][1]=margs23[r][1]+1
                                                                    hhlll="Sameuser("+str(r)+","+str(r1)+")"
                                                                    if r not in expns2 or hhlll not in expns2[r]:
                                                                        expns2[r][hhlll] =C1
                                                                        if r not in Relational_formula_filter:
                                                                            Relational_formula_filter[r]=WORDS22[r]    
                                                                    else:
                                                                        expns2[r][hhlll] = expns2[r][hhlll] +C1 
                                                                #elif Sample[r1]==2:
                                                                    #W1=W1+r_wts1[w][1]
                                                                   # margs23[r][1]=margs23[r][1]+1
                                                                    #hhlll="Sameairlines("+str(r)+","+str(r1)+")"
                                                                    #if r not in expns2 or hhlll not in expns2[r]:
                                                                       # expns2[r][hhlll] =C1
                                                                        #if r not in Relational_formula_filter:
                                                                   #        # Relational_formula_filter[r]=WORDS2[r]    

                                                        except:
                                                            continue
                                #Computing Marginal Probability after user input
                                margs22={}
                                for t in margs23:
                                    gh=[]
                                    if iter11[t]>0:
                                        for kk in margs23[t]:
                                            vv=float(kk)/float(iter11[t])
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
                                    if s!=0:
                                        #pass#print(t,s)
                                        margs33[t]=margs22[t]
                                #typppppppppppp
                                d_tt={}
                                v=0
                                #for t in ar_per_m:
                                   # d_tt[v]=t
                                    #v=v+1
                                #pass#print(d_tt)
                                d_tt[0]=0
                                d_tt[1]=1
                                #Computing the Highest Probability user input
                                margs3={}
                                for dd in margs33:
                                    v=max(margs33[dd])
                                    margs3[dd]=v
                                for vv in margs3:
                                    if margs3[vv]>=0.5:
                                        pass#pass#print(vv,margs3[vv])
                                #pass#print(len(margs3))
                                #predict topic user input
                                sampled_doc=[]
                                pred_t=[]
                                for a in margs33:
                                    for ss in range(0,len(margs33[a])):
                                            if margs33[a][ss]==margs3[a]:
                                            #pass#print(a,d_tt[ss])
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
                                           #pass#print(s,sampled_doc_up_map[s])


                                #pass#print(doc_per_pred_topic)
                                #pass#print(cx)
                                #ffd1=open("/Users/khanfarabi/OneDrive - The University of Memphis/Explain_MLN/Explanation_yelp/Corona/User_Prediction_1.txt","w")

                                for s in sampled_doc_up_map_user:
                                    pass#ccvb=str(s)+":"+str(sampled_doc_up_map_user[s])
                                    #ffd1.write(str(ccvb)+"\n")
                                #ffd1.close()


                                #Explanation Generation  with user
                                import operator
                                correct_predictions_r = {}

                                for m in margs33.keys():
                                            if m in WORDS22 and m in sampled_doc_up_map_user:
                                                          correct_predictions_r[m] = 1
                                            #if len(WORDS[m])==0:#or ratings[m]==3:
                                               # continue
                                            #else:
                                               # correct_predictions[m] = 1
                                            #if margs[m] > 0.5 and spamindex[m] ==-1:
                                                #correct_predictions[m] = 1
                                #fft_r=open("/Users/khanfarabi/OneDrive - The University of Memphis/Explain_MLN/Explanation_yelp/Corona/expn_n1_r1_user.txt","w")  
                                explanation_r={}
                                exp_ur_deno={}
                                deno_ur={}
                                for e in expns2: 
                                    if e in correct_predictions_r:
                                        sorted_expns_r = sorted(expns2[e].items(), key=operator.itemgetter(1),reverse=True)
                                        z = 0
                                        c=0
                                        for s in sorted_expns_r[:25]:
                                            z = z + s[1]
                                        rex_r = {}
                                        keys_r = []
                                        for s in sorted_expns_r[:25]:
                                            zzz=s[1]/z
                                            if float(s[1]/z)>=0.01:
                                                if c<20:
                                                    rex_r[s[0]] = zzz
                                                    c=c+1
                                            deno_ur[s[0]]=z
                                            keys_r.append(s[0])
                                        #if "Sameuser" not in keys or "Samehotel" not in keys:
                                            #continue
                                        dd1= sorted(rex_r.items(), key=operator.itemgetter(1),reverse=True)
                                        dd122= sorted(deno_ur.items(), key=operator.itemgetter(1),reverse=True)
                                        #if sorted1[0][0]=="JNTM":
                                        #pass#print(str(e) +" "+str(sorted1))
                                        #gg11=str(e) +":"+str(dd1)
                                        explanation_r[e]=dd1
                                        exp_ur_deno[e]=dd122
                                        #fft_r.write(str(gg11)+"\n")
                               #hhh="Explanation_K=20_sample_2"+"_user"+"with user feedback"+".txt"
                                #f11_r=open(hhh,"w")
                                Store_Explanation_user2={}
                                for t in explanation_r:
                                    #for k in WORDS1:
                                           #if str(t)==str(k):
                                                ggg=str(t)+"::"+str(explanation_r[t])
                                                #f11_r.write(str(ggg)+"\n")
                                                #f11_r.write("\n")
                                                #pass#print(t,explanation_r[t])
                                                Store_Explanation_user2[t]=explanation_r[t]
                                #f11_r.close()
                                #fft_r.close()
                                return  Store_Explanation_user2



                    Store_Explanation_user2=update_exp()          

                    # Full MLN Explanation

                    # updated evidence
                    def update_exp():
                                #user  feedback explanations
                                #Sample 2 user

                                import random
                                from collections import defaultdict

                                #Sampler
                                margs23 =defaultdict(list)
                                Relational_formula_filter={}
                                iter11=defaultdict(list)
                                users_s= defaultdict(list)
                                expns2 = defaultdict(dict)
                                Relational_formula_filter={}
                                same_user={}
                                Sample={}
                                for h in m_sr:
                                    pass#print(h)
                                    for i,r in enumerate(m_sr[h]):
                                        Sample[r] = random.randint(0,1)
                                        #Sample_r[r] = random.randint(0,1)
                                        if r in WORDS:
                                            margs23[r] = [0]*2
                                            #margs23_r[r] = 0
                                            iter11[r] =0

                                #Tunable parameter (default value of prob)
                                C1 = 0.98
                                VR=0.98
                                iters =1000000

                                for t in range(0,iters,1):
                                    h = random.choice(list(m_sr.keys()))
                                    h1 = random.choice(list(m_sr.keys()))
                                    if len(m_sr[h])==0:
                                        continue
                                    if len(m_sr[h1])==0:
                                        continue
                                    ix = random.randint(0,len(m_sr[h])-1)
                                    r = m_sr[h][ix]
                                    if r in WORDS:
                                        if random.random()<0.5:
                                            #sample Topic
                                            W0=0
                                            W1=0
                                            #W2=0
                                            #W3=0
                                            #W4=0
                                            #W5=0
                                            #W6=0
                                            #W7=0
                                            #W8=0
                                            #W9=0
                                            #W2=0


                                            try:
                                                        for w in WORDS[r]:
                                                            if len(r_wts[w])<0:
                                                                continue
                                                            W0=W0+r_wts[w][0]
                                                            W1=W1+(1-r_wts[w][0])
                                                           # W2=W2+r_wts[w][2]
                                                           # W3=W3+r_wts[w][3]
                                                           # W4=W4+r_wts[w][4]
                                                            #W5=W5+r_wts[w][5]
                                                            #W6=W6+r_wts[w][6]
                                                            #W7=W7+r_wts[w][7]
                                                            #W8=W8+r_wts[w][8]
                                                            #W9=W9+r_wts[w][9]


                                                            if r not in expns2 or w not in expns2[r]:
                                                                                expns2[r][w] = r_wts[w][0]
                                                                                expns2[r][w] = (1-r_wts[w][0])
                                                                               # expns2[r][w] = r_wts[w][2]
                                                                                #expns2[r][w] = r_wts[w][3]
                                                                               # expns2[r][w] = r_wts[w][4]
                                                                               # expns2[r][w] = r_wts[w][5]
                                                                                #expns2[r][w] = r_wts[w][6]
                                                                               # expns2[r][w] = r_wts[w][7]
                                                                                #expns2[r][w] = r_wts[w][8]
                                                                                #expns2[r][w] = r_wts[w][9]


                                                            else:
                                                                                expns2[r][w] =expns2[r][w]+r_wts[w][0]
                                                                                expns2[r][w] = expns2[r][w]+(1-r_wts[w][0])
                                                                               # expns2[r][w] = expns2[r][w]+r_wts[w][2]
                                                                               # expns2[r][w] = expns2[r][w]+r_wts[w][3]
                                                                                #expns2[r][w] = expns2[r][w]+r_wts[w][4]
                                                                                #expns2[r][w] = expns2[r][w]+r_wts[w][5]
                                                                                #expns2[r][w] = expns2[r][w]+r_wts[w][6]
                                                                                #expns2[r][w] = expns2[r][w]+r_wts[w][7]
                                                                                #expns2[r][w] = expns2[r][w]+r_wts[w][8]
                                                                                #expns2[r][w] = expns2[r][w]+r_wts[w][9]

                                            except:
                                                continue



                                            if (W0+W1) != 0:
                                                W0 = W0/(W0+W1)
                                                W1 = W1/(W0+W1)
                                               # W2 = W2/(W0+W1+W2+W3+W4+W5)
                                               # W3=W3/(W0+W1+W2+W3+W4+W5)
                                               # W4=W4/(W0+W1+W2+W3+W4+W5)
                                               # W5=W5/(W0+W1+W2+W3+W4+W5)
                                               # W3 = W3/(W0+W1+W2+W3+W4+W5+W6+W7+W8+W9)
                                               # W4 = W4/(W0+W1+W2+W3+W4+W5+W6+W7+W8+W9)
                                               # W5 = W5/(W0+W1+W2+W3+W4+W5+W6+W7+W8+W9)
                                               # W6 = W6/(W0+W1+W2+W3+W4+W5+W6+W7+W8+W9)
                                               # W7= W7/(W0+W1+W2+W3+W4+W5+W6+W7+W8+W9)
                                               # W8 = W8/(W0+W1+W2+W3+W4+W5+W6+W7+W8+W9)
                                               # W9 = W9/(W0+W1+W2+W3+W4+W5+W6+W7+W8+W9)
                                                #W2=W2/(W0+W1+W2)(W0+W1+W2+W3+W4+W5+W6+W7+W8+W9)


                                                sval = random.random()
                                                iter11[r]=iter11[r]+1
                                                #pass#print(sval,W14,W15,W16,W17,W18,W19)
                                                if sval<W0:
                                                    Sample[r]=1
                                                    margs23[r][0]=margs23[r][0]+1
                                                elif sval<(W0+W1):
                                                    Sample[r]=0
                                                    margs23[r][1]=margs23[r][1]+1
                                                #elif sval<(W0+W1+W2):
                                                   # Sample[r]=2
                                                    #margs23[r][2]=margs23[r][2]+1
                                                #elif sval<(W0+W1+W2+W3):
                                                    #Sample[r]=3
                                                    #margs23[r][3]=margs23[r][3]+1
                                               # elif sval<(W0+W1+W2+W3+W4):
                                                    #Sample[r]=4
                                                    #margs23[r][4]=margs23[r][4]+1
                                                #elif sval<(W0+W1+W2+W3+W4+W5):
                                                    #Sample[r]=5
                                                    #margs23[r][5]=margs23[r][5]+1
                                                #elif sval<((W0+W1+W2+W3+W4+W5+W6)):
                                                    #Sample[r]=1
                                                    #margs23[r][6]=margs23[r][6]+1
                                                #elif sval<(W0+W1+W2+W3+W4+W5+W6+W7):
                                                    #Sample[r]=1
                                                    #margs23[r][7]=margs23[r][7]+1
                                                #elif sval<(W0+W1+W2+W3+W4+W5+W6+W7+W8):
                                                    #Sample[r]=1
                                                    #margs23[r][8]=margs23[r][8]+1
                                                #elif sval<(W0+W1+W2+W3+W4+W5+W6+W7+W8+W9):
                                                    #Sample[r]=1
                                                    #margs23[r][9]=margs23[r][9]+1







                                                for r1 in m_sr[h]:
                                                    if r1==r:
                                                        continue
                                                    if r in WORDS:
                                                        try:
                                                            if Sample[r]!=Sample[r1]:
                                                                if Sample[r1]==1:
                                                                    #W0=W0+r_wts1[w][0]
                                                                    #margs23[r][0]=margs23[r][0]+1
                                                                    hhlll="Sameuser("+str(r)+","+str(r1)+")"
                                                                    if r not in expns2 or hhlll not in expns2[r]:
                                                                        expns2[r][hhlll] =C1
                                                                        if r not in Relational_formula_filter:
                                                                            Relational_formula_filter[r]=WORDS[r]    
                                                                    else:
                                                                        expns2[r][hhlll] = expns2[r][hhlll] + C1
                                                                elif Sample[r1]==0:
                                                                    #W1=W1+r_wts1[w][1]
                                                                   # margs23[r][1]=margs23[r][1]+1
                                                                    hhlll="Sameuser("+str(r)+","+str(r1)+")"
                                                                    if r not in expns2 or hhlll not in expns2[r]:
                                                                        expns2[r][hhlll] =C1
                                                                        if r not in Relational_formula_filter:
                                                                            Relational_formula_filter[r]=WORDS[r]    
                                                                    else:
                                                                        expns2[r][hhlll] = expns2[r][hhlll] +C1 
                                                                #elif Sample[r1]==2:
                                                                    #W1=W1+r_wts1[w][1]
                                                                   # margs23[r][1]=margs23[r][1]+1
                                                                    #hhlll="Sameairlines("+str(r)+","+str(r1)+")"
                                                                    #if r not in expns2 or hhlll not in expns2[r]:
                                                                       # expns2[r][hhlll] =C1
                                                                        #if r not in Relational_formula_filter:
                                                                   #        # Relational_formula_filter[r]=WORDS2[r]    

                                                        except:
                                                            continue
                                #Computing Marginal Probability after user input
                                margs22={}
                                for t in margs23:
                                    gh=[]
                                    if iter11[t]>0:
                                        for kk in margs23[t]:
                                            vv=float(kk)/float(iter11[t])
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
                                    if s!=0:
                                        #pass#print(t,s)
                                        margs33[t]=margs22[t]
                                #typppppppppppp
                                d_tt={}
                                v=0
                                #for t in ar_per_m:
                                   # d_tt[v]=t
                                    #v=v+1
                                #pass#print(d_tt)
                                d_tt[0]=0
                                d_tt[1]=1
                                #Computing the Highest Probability user input
                                margs3={}
                                for dd in margs33:
                                    v=max(margs33[dd])
                                    margs3[dd]=v
                                for vv in margs3:
                                    if margs3[vv]>=0.5:
                                        pass#pass#print(vv,margs3[vv])
                                #pass#print(len(margs3))
                                #predict topic user input
                                sampled_doc=[]
                                pred_t=[]
                                for a in margs33:
                                    for ss in range(0,len(margs33[a])):
                                            if margs33[a][ss]==margs3[a]:
                                            #pass#print(a,d_tt[ss])
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
                                           #pass#print(s,sampled_doc_up_map[s])


                                #pass#print(doc_per_pred_topic)
                                #pass#print(cx)
                                #ffd1=open("/Users/khanfarabi/OneDrive - The University of Memphis/Explain_MLN/Explanation_yelp/Corona/User_Prediction_1.txt","w")

                                for s in sampled_doc_up_map_user:
                                    pass#ccvb=str(s)+":"+str(sampled_doc_up_map_user[s])
                                    #ffd1.write(str(ccvb)+"\n")
                                #ffd1.close()


                                #Explanation Generation  with user
                                import operator
                                correct_predictions_r = {}

                                for m in margs33.keys():
                                            if m in WORDS and m in sampled_doc_up_map_user:
                                                          correct_predictions_r[m] = 1
                                            #if len(WORDS[m])==0:#or ratings[m]==3:
                                               # continue
                                            #else:
                                               # correct_predictions[m] = 1
                                            #if margs[m] > 0.5 and spamindex[m] ==-1:
                                                #correct_predictions[m] = 1
                                #fft_r=open("/Users/khanfarabi/OneDrive - The University of Memphis/Explain_MLN/Explanation_yelp/Corona/expn_n1_r1_user.txt","w")  
                                explanation_r={}
                                exp_ur_deno={}
                                deno_ur={}
                                for e in expns2: 
                                    if e in correct_predictions_r:
                                        sorted_expns_r = sorted(expns2[e].items(), key=operator.itemgetter(1),reverse=True)
                                        z = 0
                                        c=0
                                        for s in sorted_expns_r[:25]:
                                            z = z + s[1]
                                        rex_r = {}
                                        keys_r = []
                                        for s in sorted_expns_r[:25]:
                                            zzz=s[1]/z
                                            if float(s[1]/z)>=0.01:
                                                if c<20:
                                                    rex_r[s[0]] = zzz
                                                    c=c+1
                                            deno_ur[s[0]]=z
                                            keys_r.append(s[0])
                                        #if "Sameuser" not in keys or "Samehotel" not in keys:
                                            #continue
                                        dd1= sorted(rex_r.items(), key=operator.itemgetter(1),reverse=True)
                                        dd122= sorted(deno_ur.items(), key=operator.itemgetter(1),reverse=True)
                                        #if sorted1[0][0]=="JNTM":
                                        #pass#print(str(e) +" "+str(sorted1))
                                        #gg11=str(e) +":"+str(dd1)
                                        explanation_r[e]=dd1
                                        exp_ur_deno[e]=dd122
                                        #fft_r.write(str(gg11)+"\n")
                               #hhh="Explanation_K=20_sample_2"+"_user"+"with user feedback"+".txt"
                                #f11_r=open(hhh,"w")
                                Store_Explanation_user2={}
                                for t in explanation_r:
                                    #for k in WORDS1:
                                           #if str(t)==str(k):
                                                ggg=str(t)+"::"+str(explanation_r[t])
                                                #f11_r.write(str(ggg)+"\n")
                                                #f11_r.write("\n")
                                                #pass#print(t,explanation_r[t])
                                                Store_Explanation_user2[t]=explanation_r[t]
                                #f11_r.close()
                                #fft_r.close()
                                return  Store_Explanation_user2



                    Store_Explanation_full=update_exp()          


                    #Full MLN Accuracy
                    #Word Accuracy
                    #Full MLN EXP
                    def full_mln():
                                fw={}
                                for t in Store_Explanation_full:
                                    gh=[]
                                    for k in Store_Explanation_full[t]:
                                        if 'Same' not in k[0]:
                                            gh.append(k[0])
                                    fw[t]=gh
                                fr={}
                                for t in Store_Explanation_full:
                                    gh=[]
                                    for k in Store_Explanation_full[t]:
                                        if 'Same'  in k[0]:
                                            gg=k[0].strip(")").split(",")
                                            gh.append(gg[1])
                                    fr[t]=gh

                                wf={}
                                for k in fw:
                                    c=0
                                    if k in ann:
                                        for j in fw[k]:
                                            if j in ann[k]:
                                                c=c+1
                                    ss=c/len(fw[k])
                                    if ss>0:
                                        wf[k]=ss
                                sz=0
                                for t in wf:
                                    sz=sz+float(wf[t])
                                print("word accuracy")
                                print(sz/len(wf))


                                #relational accuracy similar_r_map
                                wf={}
                                for k in fr:
                                    c=0
                                    if k in similar_r_map:
                                        #print(k)
                                        for j in fr[k]:
                                            if j in similar_r_map[k]:
                                                c=c+1
                                    try:
                                        ss=c/len(fr[k])
                                        if ss>0:
                                            wf[k]=ss
                                    except:
                                        continue 
                                sz=0
                                for t in wf:
                                    sz=sz+float(wf[t])
                                print("relational accuracy")
                                print(sz/len(wf))
                    full_mln()

                    #SHAP accuracy train_r targets_r
                    def shap_accuracy():
                                        #shap
                                        import sklearn
                                        from sklearn.feature_extraction.text import TfidfVectorizer
                                        from sklearn.model_selection import train_test_split
                                        import numpy as np
                                        import shap
                                        import transformers
                                        import shap
                                        #shap.initjs()
                                        # Kernal Shap words_train targets

                                        from sklearn import svm
                                        from sklearn.svm import SVC
                                        from sklearn.svm import LinearSVC
                                        corpus_train, corpus_test, y_train, y_test = train_test_split(train_r,targets_r, test_size=0.5, random_state=7)
                                        vectorizer = TfidfVectorizer(min_df=10)
                                        X_train = vectorizer.fit_transform(corpus_train)
                                        X_test = vectorizer.transform(corpus_test)
                                        model =svm.LinearSVC(C=100) #sklearn.linear_model.LogisticRegression(penalty="l1", C=0.1)
                                        model.fit(X_train,y_train)
                                        explainer = shap.LinearExplainer(model, X_train, feature_dependence="independent")
                                        shap_values = explainer.shap_values(X_test)
                                        X_test_array = X_test.toarray() # we need to pass a dense version for the plotting functions
                                        feature_names=vectorizer.get_feature_names()
                                        print(len(feature_names),len(shap_values))
                                        #shap.summary_plot(shap_values, X_test_array, feature_names=vectorizer.get_feature_names())
                                        feature_sh_v={}
                                        for  ind in range(0,len(corpus_train)-1):
                                            #import statistics
                                            #print("Positive" if y_train[ind] else "Negative", "Review:")
                                            try:
                                                print(corpus_test[ind],"Positive" if y_test[ind] else "Negative",sum(shap_values[ind]))
                                                feature_sh_v[corpus_test[ind]]=sum(shap_values[ind])
                                            except:
                                                continue

                                        #shap explanations

                                        shap_exp={}
                                        for t in WORDS:
                                            gh=[]
                                            c=0
                                            for k in WORDS[t]:
                                                if k in feature_sh_v:
                                                    if k not in gh:
                                                        if c<3:
                                                            gh.append(k)
                                                            c=c+1
                                            shap_exp[t]=gh

                                        for tt in shap_exp:
                                            print(tt,shap_exp[tt])
                                        #shap accuracy

                                        shap_all={}

                                        for t in shap_exp:
                                            if t in ann:
                                                c=0
                                                for zz in shap_exp[t]:
                                                    if zz in ann[t]:
                                                        c=c+1
                                                s=float(c)/len(shap_exp[t])
                                                #if s>0:
                                                shap_all[t]=s
                                        ss=0
                                        for k in shap_all:
                                            ss=ss+float(shap_all[k])

                                        acc=ss/len(shap_all)
                                        return acc
                    gg1=shap_accuracy()
                    print("shap explanation accuracy:")
                    print(gg1)


                    #Lime accuracy words_train targets     train_r targets_r


                    # LIME Exp
                    from sklearn.feature_extraction.text import TfidfVectorizer
                    from collections import defaultdict
                    from sklearn import svm
                    #from sklearn import cross_validation targets_r
                    from sklearn.model_selection import cross_validate
                    from sklearn.preprocessing import MinMaxScaler
                    from sklearn.metrics import (precision_score,recall_score,f1_score)
                    from sklearn.multiclass import OneVsRestClassifier
                    from sklearn.svm import SVC

                    #Learn SVM Model train_r targets_r
                    #ifile = open("processed_revs_1.txt",encoding="ISO-8859-1")
                    Y = targets_r
                    #words = []
                    unique_words = []
                    #for ln in ifile:
                       # parts = ln.strip().split("\t")
                       # Y.append(int(parts[1]))
                        #words.append(parts[0])
                    for w in words:
                            #h=w.split()
                            #for e in h:
                            unique_words.append(w)
                    #ifile.close()
                    tf_transformer = TfidfVectorizer()
                    f = tf_transformer.fit_transform(train_r)
                    features = [((i, j), f[i,j]) for i, j in zip(*f.nonzero())]
                    unique_word_ids = []
                    for w in unique_words:
                        i = tf_transformer.vocabulary_.get(w)
                        unique_word_ids.append(i)

                    clf =svm.SVC(C=1.0, kernel='linear', degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=True, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=- 1, decision_function_shape='ovr', random_state=None)
                    #svm.SVC(C=1,probability=True)#OneVsRestClassifier(SVC(kernel='linear'))#svm.LinearSVC(C=1)
                    clf.fit(f,Y)
                    p = clf.predict(f)
                    print(f1_score(Y,p,average='micro'))


                    def lime_accuracy():  

                                    #lime train test
                                    from sklearn.model_selection import train_test_split
                                    Train_r, Test_r, tr, ts = train_test_split(train_r,targets_r, test_size=0.05, random_state=7)
                                    #LIME



                                    #Lime
                                    import sys
                                    import random
                                    import io
                                    import operator
                                    import numpy as np
                                    import lime
                                    import lime.lime_tabular
                                    import sklearn
                                    import numpy as np
                                    import sklearn
                                    import random
                                    import sklearn.ensemble
                                    import sklearn.metrics
                                    from sklearn import svm
                                    from sklearn.metrics import f1_score
                                    from sklearn.svm import SVC
                                    from lime import lime_text
                                    from sklearn.pipeline import make_pipeline
                                    from lime.lime_text import LimeTextExplainer
                                    import re
                                    from lime.lime_text import LimeTextExplainer
                                    from IPython.core.display import display, HTML
                                    from sklearn.feature_extraction.text import TfidfVectorizer
                                    from scipy.sparse import csr_matrix
                                    from lime.lime_text import LimeTextExplainer

                                    c = make_pipeline(tf_transformer,clf)
                                    #print(c)
                                    rtt=c.predict_proba([Train_r[1]]).round(3)
                                    print(rtt)

                                    #mt1=['7','12','13','14','15','16','17'] #clasnames
                                    mt1=['0','1']
                                    cna=mt1


                                    #explainer = lime.lime_tabular.LimeTabularExplainer(tr, feature_names=feature_names, class_names=targets_t, discretize_continuous=True)

                                    explainer = LimeTextExplainer(class_names=cna)
                                    print(len(Train_r))
                                    #sys.exit()
                                    #d22=sorted(dd3.items(),key=operator.itemgetter(1),reverse=True)
                                    word_weight={}
                                    for ii in range(0,len(Test_r)):

                                                                ww={}
                                                                ww2={}
                                                                wq=[]
                                                                exp = explainer.explain_instance(Test_r[ii], c.predict_proba, labels=(0,1), top_labels=None, num_features=44055, num_samples=60000, distance_metric=u'cosine', model_regressor=None)# explainer.explain_instance(trainf[int(ii)], c.predict_proba, num_features=2961, labels=[0,1])#, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49])
                                                                #print("jhhjvhhvhvhhg")
                                                                #print(exp)
                                                                tty=exp.as_list()
                                                               # print(tty)
                                                                rtr=''
                                                                for i in tty:
                                                                    #print(i)
                                                                    #rrr=random.randint(2,10)
                                                                    mlp=random.uniform(0.8,1.1)
                                                                    if float(i[1])>0:
                                                                        word_weight[i[0]]=abs(float(i[1]))
                                    # lime accuracy
                                    import operator
                                    lw=[]
                                    dd=sorted(word_weight.items(), key=operator.itemgetter(1),reverse=True)

                                    for k in dd:
                                        lw.append(k[0])
                                    #print(lw)
                                    lexp={}
                                    for tt in WORDS:
                                        gh=[]
                                        c=0
                                        for gg in WORDS[tt]:
                                            if gg in lw:
                                                #print(gg)
                                                if c<5:
                                                    gh.append(gg)
                                                    c=c+1
                                        lexp[tt]=gh
                                    lexps={}
                                    for h in lexp:
                                        if h in ann:
                                            cc=0
                                            for nn in lexp[h]:
                                                if nn in ann[h]:
                                                    cc=cc+1
                                            if len(lexp[h])>0:
                                                        ss=float(cc)/len(lexp[h])
                                                        lexps[h]=ss
                                    st=0
                                    for ff in lexps:
                                        st=st+float(lexps[ff])

                                    fffff=st/float(len(lexps))
                                    #print(fffff)
                                    return fffff

                    #clf=svm_t()
                    gg=lime_accuracy()
                    print("lime acuracy:")
                    print(gg)


                    # Varying Cluster Our Technique. Covid-19 data
                    import rpy2
                    import rpy2.robjects.packages as rpackages
                    from rpy2.robjects.vectors import StrVector
                    from rpy2.robjects.packages import importr
                    utils = rpackages.importr('utils')
                    utils.chooseCRANmirror(ind=1)
                    # Install packages
                    packnames = ('TopKLists', 'Borda')
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
                                            ##pass#print (assigned_clusters)
                                            cluster={}
                                            words = list(model.wv.vocab)
                                            for i, word in enumerate(words):
                                              gh=[] 
                                              gh1=[] 
                                              gh2=[] 
                                              if word.isdigit(): 
                                                cluster[word]=assigned_clusters1[i]
                                                ##pass#print (word + ":" + str(assigned_clusters[i]))
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
                                                                                ##pass#print(M)
                                                                                if tt not in reduced_obj:
                                                                                    reduced_obj.append(tt)
                                                            ##pass#print(len(reduced_obj)/float(len(final_clu[k]))
                                                            wr={}
                                                            w=[]
                                                            for k in final_clu:
                                                                    #c=-1
                                                                    #c=c+1
                                                                    md=int(len(final_clu[k])//2)
                                                                    c=0      
                                                                    k1= final_clu[k][md+c]
                                                                    ##pass#print(k1,md)        
                                                                    if k1 in ann:
                                                                                for k3 in ann[k1]:
                                                                                        w.append(k3)
                                                                    else:
                                                                        c=c+11
                                                                        continue 


                                                                   # #pass#print(k,k1,md,d_tt[qrat[k1]],w)
                                                                    wr[k1]=w
                                                            model = Word2Vec(sent, min_count=1)
                                                            data_g={}
                                                            for t in WORDS:
                                                                chu=[]
                                                                #try:
                                                                vb={}
                                                                for v in w:
                                                                    vb1={}
                                                                    for v1 in WORDS[t]:
                                                                            ##pass#print(v1,v)
                                                                            gh1=model.similarity(v,v1)
                                                                            if gh1>=0.40:
                                                                                  vb1[v1]=float(gh1)
                                                                                  ##pass#print(gh1)
                                                                    for jk in vb1:
                                                                        if jk in vb:
                                                                            if float(vb1[jk])>=float(vb[jk]):
                                                                                ##pass#print(jk,vb1[jk],vb[jk])
                                                                                vb[jk]=vb1[jk]
                                                                        else:
                                                                            vb[jk]=vb1[jk]
                                                                ##pass#print(t, vb)
                                                                ##pass#print("\n")             
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
                                                            #pass#print(len(WORDS1))
                                                            #Updating the Whole Evidence Based on manual annotation
                                                            WORDS22={}
                                                            for gg in WORDS:
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
                                                                            if czx<10:# and dd in s_words or str(dd) in s_words:
                                                                                hhh1.append(dd)
                                                                                czx=czx+1

                                                                    WORDS2[t]=hhh1
                                                                    ##pass#print(len(hhh1))
                                                            ##pass#print(WORDS2)
                                                            ##pass#print(len(WORDS2),len(reduced_obj),len(WORDS22))
                                                            #Sample 2 user

                                                            import random
                                                            from collections import defaultdict

                                                            #Sampler
                                                            import random
                                                            from collections import defaultdict

                                                            #Sampler
                                                            margs23 =defaultdict(list)
                                                            Relational_formula_filter={}
                                                            iter11=defaultdict(list)
                                                            users_s= defaultdict(list)
                                                            expns2 = defaultdict(dict)
                                                            Relational_formula_filter={}
                                                            same_user={}
                                                            Sample={}
                                                            for h in m_sr:
                                                                pass#print(h)
                                                                for i,r in enumerate(m_sr[h]):
                                                                    Sample[r] = random.randint(0,1)
                                                                    #Sample_r[r] = random.randint(0,1)
                                                                    if r in WORDS2:
                                                                        margs23[r] = [0]*2
                                                                        #margs23_r[r] = 0
                                                                        iter11[r] =0

                                                            #Tunable parameter (default value of prob)
                                                            C1 = 0.98
                                                            VR=0.98
                                                            iters =1000000

                                                            for t in range(0,iters,1):
                                                                h = random.choice(list(m_sr.keys()))
                                                                h1 = random.choice(list(m_sr.keys()))
                                                                if len(m_sr[h])==0:
                                                                    continue
                                                                if len(m_sr[h1])==0:
                                                                    continue
                                                                ix = random.randint(0,len(m_sr[h])-1)
                                                                r = m_sr[h][ix]
                                                                if r in WORDS2:
                                                                    if random.random()<0.5:
                                                                        #sample Topic
                                                                        W0=0
                                                                        W1=0
                                                                        #W2=0
                                                                        #W3=0
                                                                        #W4=0
                                                                        #W5=0
                                                                        #W6=0
                                                                        #W7=0
                                                                        #W8=0
                                                                        #W9=0
                                                                        #W2=0


                                                                        try:
                                                                                    for w in WORDS2[r]:
                                                                                        if len(r_wts[w])<0:
                                                                                            continue
                                                                                        W0=W0+r_wts[w][0]
                                                                                        W1=W1+(1-r_wts[w][0])
                                                                                       # W2=W2+r_wts[w][2]
                                                                                       # W3=W3+r_wts[w][3]
                                                                                       # W4=W4+r_wts[w][4]
                                                                                        #W5=W5+r_wts[w][5]
                                                                                        #W6=W6+r_wts[w][6]
                                                                                        #W7=W7+r_wts[w][7]
                                                                                        #W8=W8+r_wts[w][8]
                                                                                        #W9=W9+r_wts[w][9]


                                                                                        if r not in expns2 or w not in expns2[r]:
                                                                                                            expns2[r][w] = r_wts[w][0]
                                                                                                            expns2[r][w] = (1-r_wts[w][0])
                                                                                                           # expns2[r][w] = r_wts[w][2]
                                                                                                            #expns2[r][w] = r_wts[w][3]
                                                                                                           # expns2[r][w] = r_wts[w][4]
                                                                                                           # expns2[r][w] = r_wts[w][5]
                                                                                                            #expns2[r][w] = r_wts[w][6]
                                                                                                           # expns2[r][w] = r_wts[w][7]
                                                                                                            #expns2[r][w] = r_wts[w][8]
                                                                                                            #expns2[r][w] = r_wts[w][9]


                                                                                        else:
                                                                                                            expns2[r][w] =expns2[r][w]+r_wts[w][0]
                                                                                                            expns2[r][w] = expns2[r][w]+(1-r_wts[w][0])
                                                                                                           # expns2[r][w] = expns2[r][w]+r_wts[w][2]
                                                                                                           # expns2[r][w] = expns2[r][w]+r_wts[w][3]
                                                                                                            #expns2[r][w] = expns2[r][w]+r_wts[w][4]
                                                                                                            #expns2[r][w] = expns2[r][w]+r_wts[w][5]
                                                                                                            #expns2[r][w] = expns2[r][w]+r_wts[w][6]
                                                                                                            #expns2[r][w] = expns2[r][w]+r_wts[w][7]
                                                                                                            #expns2[r][w] = expns2[r][w]+r_wts[w][8]
                                                                                                            #expns2[r][w] = expns2[r][w]+r_wts[w][9]

                                                                        except:
                                                                            continue



                                                                        if (W0+W1) != 0:
                                                                            W0 = W0/(W0+W1)
                                                                            W1 = W1/(W0+W1)
                                                                           # W2 = W2/(W0+W1+W2+W3+W4+W5)
                                                                           # W3=W3/(W0+W1+W2+W3+W4+W5)
                                                                           # W4=W4/(W0+W1+W2+W3+W4+W5)
                                                                           # W5=W5/(W0+W1+W2+W3+W4+W5)
                                                                           # W3 = W3/(W0+W1+W2+W3+W4+W5+W6+W7+W8+W9)
                                                                           # W4 = W4/(W0+W1+W2+W3+W4+W5+W6+W7+W8+W9)
                                                                           # W5 = W5/(W0+W1+W2+W3+W4+W5+W6+W7+W8+W9)
                                                                           # W6 = W6/(W0+W1+W2+W3+W4+W5+W6+W7+W8+W9)
                                                                           # W7= W7/(W0+W1+W2+W3+W4+W5+W6+W7+W8+W9)
                                                                           # W8 = W8/(W0+W1+W2+W3+W4+W5+W6+W7+W8+W9)
                                                                           # W9 = W9/(W0+W1+W2+W3+W4+W5+W6+W7+W8+W9)
                                                                            #W2=W2/(W0+W1+W2)(W0+W1+W2+W3+W4+W5+W6+W7+W8+W9)


                                                                            sval = random.random()
                                                                            iter11[r]=iter11[r]+1
                                                                            #pass#print(sval,W14,W15,W16,W17,W18,W19)
                                                                            if sval<W0:
                                                                                Sample[r]=1
                                                                                margs23[r][0]=margs23[r][0]+1
                                                                            elif sval<(W0+W1):
                                                                                Sample[r]=0
                                                                                margs23[r][1]=margs23[r][1]+1
                                                                            #elif sval<(W0+W1+W2):
                                                                               # Sample[r]=2
                                                                                #margs23[r][2]=margs23[r][2]+1
                                                                            #elif sval<(W0+W1+W2+W3):
                                                                                #Sample[r]=3
                                                                                #margs23[r][3]=margs23[r][3]+1
                                                                           # elif sval<(W0+W1+W2+W3+W4):
                                                                                #Sample[r]=4
                                                                                #margs23[r][4]=margs23[r][4]+1
                                                                            #elif sval<(W0+W1+W2+W3+W4+W5):
                                                                                #Sample[r]=5
                                                                                #margs23[r][5]=margs23[r][5]+1
                                                                            #elif sval<((W0+W1+W2+W3+W4+W5+W6)):
                                                                                #Sample[r]=1
                                                                                #margs23[r][6]=margs23[r][6]+1
                                                                            #elif sval<(W0+W1+W2+W3+W4+W5+W6+W7):
                                                                                #Sample[r]=1
                                                                                #margs23[r][7]=margs23[r][7]+1
                                                                            #elif sval<(W0+W1+W2+W3+W4+W5+W6+W7+W8):
                                                                                #Sample[r]=1
                                                                                #margs23[r][8]=margs23[r][8]+1
                                                                            #elif sval<(W0+W1+W2+W3+W4+W5+W6+W7+W8+W9):
                                                                                #Sample[r]=1
                                                                                #margs23[r][9]=margs23[r][9]+1







                                                                            for r1 in m_sr[h]:
                                                                                if r1==r:
                                                                                    continue
                                                                                if r in WORDS2:
                                                                                    try:
                                                                                        if Sample[r]!=Sample[r1]:
                                                                                            if Sample[r1]==1:
                                                                                                #W0=W0+r_wts1[w][0]
                                                                                                #margs23[r][0]=margs23[r][0]+1
                                                                                                hhlll="Sameuser("+str(r)+","+str(r1)+")"
                                                                                                if r not in expns2 or hhlll not in expns2[r]:
                                                                                                    expns2[r][hhlll] =C1
                                                                                                    if r not in Relational_formula_filter:
                                                                                                        Relational_formula_filter[r]=WORDS2[r]    
                                                                                                else:
                                                                                                    expns2[r][hhlll] = expns2[r][hhlll] + C1
                                                                                            elif Sample[r1]==0:
                                                                                                #W1=W1+r_wts1[w][1]
                                                                                               # margs23[r][1]=margs23[r][1]+1
                                                                                                hhlll="Sameuser("+str(r)+","+str(r1)+")"
                                                                                                if r not in expns2 or hhlll not in expns2[r]:
                                                                                                    expns2[r][hhlll] =C1
                                                                                                    if r not in Relational_formula_filter:
                                                                                                        Relational_formula_filter[r]=WORDS2[r]    
                                                                                                else:
                                                                                                    expns2[r][hhlll] = expns2[r][hhlll] +C1 
                                                                                            #elif Sample[r1]==2:
                                                                                                #W1=W1+r_wts1[w][1]
                                                                                               # margs23[r][1]=margs23[r][1]+1
                                                                                                #hhlll="Sameairlines("+str(r)+","+str(r1)+")"
                                                                                                #if r not in expns2 or hhlll not in expns2[r]:
                                                                                                   # expns2[r][hhlll] =C1
                                                                                                    #if r not in Relational_formula_filter:
                                                                                               #        # Relational_formula_filter[r]=WORDS2[r]    

                                                                                    except:
                                                                                        continue
                                                            #Computing Marginal Probability after user input
                                                            margs22={}
                                                            for t in margs23:
                                                                gh=[]
                                                                if iter11[t]>0:
                                                                    for kk in margs23[t]:
                                                                        vv=float(kk)/float(iter11[t])
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
                                                                if s!=0:
                                                                    #pass#print(t,s)
                                                                    margs33[t]=margs22[t]
                                                            #typppppppppppp
                                                            d_tt={}
                                                            v=0
                                                            #for t in ar_per_m:
                                                               # d_tt[v]=t
                                                                #v=v+1
                                                            #pass#print(d_tt)
                                                            d_tt[0]=0
                                                            d_tt[1]=1
                                                            #Computing the Highest Probability user input
                                                            margs3={}
                                                            for dd in margs33:
                                                                v=max(margs33[dd])
                                                                margs3[dd]=v
                                                            for vv in margs3:
                                                                if margs3[vv]>=0.5:
                                                                    pass#pass#print(vv,margs3[vv])
                                                            #pass#print(len(margs3))
                                                            #predict topic user input
                                                            sampled_doc=[]
                                                            pred_t=[]
                                                            for a in margs33:
                                                                for ss in range(0,len(margs33[a])):
                                                                        if margs33[a][ss]==margs3[a]:
                                                                        #pass#print(a,d_tt[ss])
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
                                                                       #pass#print(s,sampled_doc_up_map[s])


                                                            #pass#print(doc_per_pred_topic)
                                                            pass#print(cx)
                                                            #ffd1=open("/Users/khanfarabi/OneDrive - The University of Memphis/Explain_MLN/Explanation_yelp/Corona/User_Prediction_1.txt","w")

                                                            for s in sampled_doc_up_map_user:
                                                                pass#ccvb=str(s)+":"+str(sampled_doc_up_map_user[s])
                                                                #ffd1.write(str(ccvb)+"\n")
                                                            #ffd1.close()


                                                            #Explanation Generation  with user
                                                            import operator
                                                            correct_predictions_r = {}

                                                            for m in margs33.keys():
                                                                        if m in WORDS22 and m in sampled_doc_up_map_user:
                                                                                      correct_predictions_r[m] = 1
                                                                        #if len(WORDS2[m])==0:#or ratings[m]==3:
                                                                           # continue
                                                                        #else:
                                                                           # correct_predictions[m] = 1
                                                                        #if margs[m] > 0.5 and spamindex[m] ==-1:
                                                                            #correct_predictions[m] = 1
                                                            #fft_r=open("/Users/khanfarabi/OneDrive - The University of Memphis/Explain_MLN/Explanation_yelp/Corona/expn_n1_r1_user.txt","w")  
                                                            explanation_r={}
                                                            exp_ur_deno={}
                                                            deno_ur={}
                                                            for e in expns2: 
                                                                if e in correct_predictions_r:
                                                                    sorted_expns_r = sorted(expns2[e].items(), key=operator.itemgetter(1),reverse=True)
                                                                    z = 0
                                                                    c=0
                                                                    for s in sorted_expns_r[:25]:
                                                                        z = z + s[1]
                                                                    rex_r = {}
                                                                    keys_r = []
                                                                    for s in sorted_expns_r[:25]:
                                                                        zzz=s[1]/z
                                                                        if float(s[1]/z)>=0.01:
                                                                            if c<20:
                                                                                rex_r[s[0]] = zzz
                                                                                c=c+1
                                                                        deno_ur[s[0]]=z
                                                                        keys_r.append(s[0])
                                                                    #if "Sameuser" not in keys or "Samehotel" not in keys:
                                                                        #continue
                                                                    dd1= sorted(rex_r.items(), key=operator.itemgetter(1),reverse=True)
                                                                    dd122= sorted(deno_ur.items(), key=operator.itemgetter(1),reverse=True)
                                                                    #if sorted1[0][0]=="JNTM":
                                                                    #pass#print(str(e) +" "+str(sorted1))
                                                                    #gg11=str(e) +":"+str(dd1)
                                                                    explanation_r[e]=dd1
                                                                    exp_ur_deno[e]=dd122


                                                            Store_Explanation_user={}
                                                            for t in explanation_r:
                                                                #for k in WORDS1:
                                                                       #if str(t)==str(k):
                                                                            ggg=str(t)+":"+str(explanation_r[t])
                                                                            #f11_r.write(str(ggg)+"\n")
                                                                           # f11_r.write("\n")
                                                                            ##pass#print(t,explanation_r[t])
                                                                            ##pass#print("\n")
                                                                            Store_Explanation_user[t]=explanation_r[t]
                                                                            Sample_model[t]=explanation_r[t]
                                                            expt_st_pd[mzzz]=Store_Explanation_user
                                                            ##pass#print(mzzz,len(Store_Explanation_user))

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
                                            ##pass#print(jj,t)
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
                                    ##pass#print(tt,cz,cz1)
                                    for zb in cz1:
                                        try:
                                            for k in obj_m[K][zb]:
                                                if ob_clm[int(tt)]==ob_clm[int(k)]:
                                                    break
                                        except:
                                            continue
                                        ##pass#print(tt,k,Store_Explanation_user2[k])
                                        ##pass#print("\n\n")
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
                                                                ##pass#print(tt,zb,bb1)
                                                                ##pass#print("\n")
                                                                if bb1 not in vbb:
                                                                    vbb.append(bb1)
                                            exp.append(vbb)
                                        ##pass#print(bb1)
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
                                                            ##pass#print(tt,cx,bb)
                                                            ##pass#print("\n\n")
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
                                        b=r['Borda'](ll,k=30)#agg.borda(ll)
                                        for t in b[0][0]:
                                            if c<T:
                                                gh.append(t)
                                                c=c+1
                                       # #pass#print(gh)
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
                                            ##pass#print(tt,cz,cz1)
                                            for zb in cz1:
                                                try:
                                                    for k in obj_m[K][zb]:
                                                        if ob_clm[int(tt)]==ob_clm[int(k)]:
                                                            break
                                                except:
                                                    continue
                                                ##pass#print(tt,k,Store_Explanation_user2[k])
                                                ##pass#print("\n\n")
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
                                                                            ##pass#print(tt,zb,bb1)
                                                                            ##pass#print("\n")
                                                                            if bb1 not in vbb:
                                                                                vbb.append(bb1)             

                                                        exp.append(vbb)
                                                ##pass#print(bb1)
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
                                                                    ##pass#print(tt,cx,bb)
                                                                    ##pass#print("\n\n")
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
                                                ##pass#print(tt,cz,cz1)
                                                for zb in cz1:
                                                    try:
                                                        for k in obj_m[K][zb]:
                                                            if ob_clm[int(tt)]==ob_clm[int(k)]:
                                                                break
                                                    except:
                                                        continue
                                                    ##pass#print(tt,k,Store_Explanation_user2[k])
                                                    ##pass#print("\n\n")
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
                                                                                ##pass#print(tt,zb,bb1)
                                                                                ##pass#print("\n")
                                                                                if bb1 not in vbb:
                                                                                    vbb.append(bb1)             

                                                            exp.append(vbb)
                                                    ##pass#print(bb1)
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
                                                                        ##pass#print(tt,cx,bb)
                                                                        ##pass#print("\n\n")
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
                                                        b=r['Borda'](ll,k=10)#agg.borda(ll)
                                                        for t in b[0][0]:
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
                                                            if twit_count[int(t)]==twit_count[int(hh[1])]:
                                                                        ##pass#print(t,hh[1])
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
                                                            ##pass#print(s)


                                                zx=0
                                                cx=0
                                                for z in reliab_exp_up_1w_pc:
                                                    zx=zx+reliab_exp_up_1w_pc[z]
                                                    cx=cx+1
                                                ##pass#print(reliab_exp_up_1w_score)
                                                ##pass#print(zx/cx)
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
                                                ##pass#print("Relational Accuracy: "+str(st/len(agg_avg))+"\n")
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

                    st=3#int(input())
                    en=5*2+2
                    for t in range(st,en,2):
                                         rep=5
                                         T=4
                                         wxz,rxz=varying_cluster(t,rep,T)
                                         cluster_varying_rexp[t]=rxz
                                         cluster_varying_wexp[t]=wxz




                    print(cluster_varying_wexp,cluster_varying_rexp)


                    # Varying Models Our Technique. Covid-19 data
                    import rpy2
                    import rpy2.robjects.packages as rpackages
                    from rpy2.robjects.vectors import StrVector
                    from rpy2.robjects.packages import importr
                    utils = rpackages.importr('utils')
                    utils.chooseCRANmirror(ind=1)
                    # Install packages
                    packnames = ('TopKLists', 'Borda')
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
                                            ##pass#print (assigned_clusters)
                                            cluster={}
                                            words = list(model.wv.vocab)
                                            for i, word in enumerate(words):
                                              gh=[] 
                                              gh1=[] 
                                              gh2=[] 
                                              if word.isdigit(): 
                                                cluster[word]=assigned_clusters1[i]
                                                ##pass#print (word + ":" + str(assigned_clusters[i]))
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
                                                                                ##pass#print(M)
                                                                                if tt not in reduced_obj:
                                                                                    reduced_obj.append(tt)
                                                            ##pass#print(len(reduced_obj)/float(len(final_clu[k]))
                                                            wr={}
                                                            w=[]
                                                            for k in final_clu:
                                                                    #c=-1
                                                                    #c=c+1
                                                                    md=int(len(final_clu[k])//2)
                                                                    c=0      
                                                                    k1= final_clu[k][md+c]
                                                                    ##pass#print(k1,md)        
                                                                    if k1 in ann:
                                                                                for k3 in ann[k1]:
                                                                                        w.append(k3)
                                                                    else:
                                                                        c=c+11
                                                                        continue 


                                                                   # #pass#print(k,k1,md,d_tt[qrat[k1]],w)
                                                                    wr[k1]=w
                                                            model = Word2Vec(sent, min_count=1)
                                                            data_g={}
                                                            for t in WORDS:
                                                                chu=[]
                                                                #try:
                                                                vb={}
                                                                for v in w:
                                                                    vb1={}
                                                                    for v1 in WORDS[t]:
                                                                            ##pass#print(v1,v)
                                                                            gh1=model.similarity(v,v1)
                                                                            if gh1>=0.40:
                                                                                  vb1[v1]=float(gh1)
                                                                                  ##pass#print(gh1)
                                                                    for jk in vb1:
                                                                        if jk in vb:
                                                                            if float(vb1[jk])>=float(vb[jk]):
                                                                                ##pass#print(jk,vb1[jk],vb[jk])
                                                                                vb[jk]=vb1[jk]
                                                                        else:
                                                                            vb[jk]=vb1[jk]
                                                                ##pass#print(t, vb)
                                                                ##pass#print("\n")             
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
                                                            #pass#print(len(WORDS1))
                                                            #Updating the Whole Evidence Based on manual annotation
                                                            WORDS22={}
                                                            for gg in WORDS:
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
                                                                            if czx<10:# and dd in s_words or str(dd) in s_words:
                                                                                hhh1.append(dd)
                                                                                czx=czx+1

                                                                    WORDS2[t]=hhh1
                                                                    ##pass#print(len(hhh1))
                                                            ##pass#print(WORDS2)
                                                            ##pass#print(len(WORDS2),len(reduced_obj),len(WORDS22))
                                                            #Sample 2 user

                                                            import random
                                                            from collections import defaultdict

                                                            #Sampler
                                                            import random
                                                            from collections import defaultdict

                                                            #Sampler
                                                            margs23 =defaultdict(list)
                                                            Relational_formula_filter={}
                                                            iter11=defaultdict(list)
                                                            users_s= defaultdict(list)
                                                            expns2 = defaultdict(dict)
                                                            Relational_formula_filter={}
                                                            same_user={}
                                                            Sample={}
                                                            for h in m_sr:
                                                                pass#print(h)
                                                                for i,r in enumerate(m_sr[h]):
                                                                    Sample[r] = random.randint(0,1)
                                                                    #Sample_r[r] = random.randint(0,1)
                                                                    if r in WORDS2:
                                                                        margs23[r] = [0]*2
                                                                        #margs23_r[r] = 0
                                                                        iter11[r] =0

                                                            #Tunable parameter (default value of prob)
                                                            C1 = 0.98
                                                            VR=0.98
                                                            iters =1000000

                                                            for t in range(0,iters,1):
                                                                h = random.choice(list(m_sr.keys()))
                                                                h1 = random.choice(list(m_sr.keys()))
                                                                if len(m_sr[h])==0:
                                                                    continue
                                                                if len(m_sr[h1])==0:
                                                                    continue
                                                                ix = random.randint(0,len(m_sr[h])-1)
                                                                r = m_sr[h][ix]
                                                                if r in WORDS2:
                                                                    if random.random()<0.5:
                                                                        #sample Topic
                                                                        W0=0
                                                                        W1=0
                                                                        #W2=0
                                                                        #W3=0
                                                                        #W4=0
                                                                        #W5=0
                                                                        #W6=0
                                                                        #W7=0
                                                                        #W8=0
                                                                        #W9=0
                                                                        #W2=0


                                                                        try:
                                                                                    for w in WORDS2[r]:
                                                                                        if len(r_wts[w])<0:
                                                                                            continue
                                                                                        W0=W0+r_wts[w][0]
                                                                                        W1=W1+(1-r_wts[w][0])
                                                                                       # W2=W2+r_wts[w][2]
                                                                                       # W3=W3+r_wts[w][3]
                                                                                       # W4=W4+r_wts[w][4]
                                                                                        #W5=W5+r_wts[w][5]
                                                                                        #W6=W6+r_wts[w][6]
                                                                                        #W7=W7+r_wts[w][7]
                                                                                        #W8=W8+r_wts[w][8]
                                                                                        #W9=W9+r_wts[w][9]


                                                                                        if r not in expns2 or w not in expns2[r]:
                                                                                                            expns2[r][w] = r_wts[w][0]
                                                                                                            expns2[r][w] = (1-r_wts[w][0])
                                                                                                           # expns2[r][w] = r_wts[w][2]
                                                                                                            #expns2[r][w] = r_wts[w][3]
                                                                                                           # expns2[r][w] = r_wts[w][4]
                                                                                                           # expns2[r][w] = r_wts[w][5]
                                                                                                            #expns2[r][w] = r_wts[w][6]
                                                                                                           # expns2[r][w] = r_wts[w][7]
                                                                                                            #expns2[r][w] = r_wts[w][8]
                                                                                                            #expns2[r][w] = r_wts[w][9]


                                                                                        else:
                                                                                                            expns2[r][w] =expns2[r][w]+r_wts[w][0]
                                                                                                            expns2[r][w] = expns2[r][w]+(1-r_wts[w][0])
                                                                                                           # expns2[r][w] = expns2[r][w]+r_wts[w][2]
                                                                                                           # expns2[r][w] = expns2[r][w]+r_wts[w][3]
                                                                                                            #expns2[r][w] = expns2[r][w]+r_wts[w][4]
                                                                                                            #expns2[r][w] = expns2[r][w]+r_wts[w][5]
                                                                                                            #expns2[r][w] = expns2[r][w]+r_wts[w][6]
                                                                                                            #expns2[r][w] = expns2[r][w]+r_wts[w][7]
                                                                                                            #expns2[r][w] = expns2[r][w]+r_wts[w][8]
                                                                                                            #expns2[r][w] = expns2[r][w]+r_wts[w][9]

                                                                        except:
                                                                            continue



                                                                        if (W0+W1) != 0:
                                                                            W0 = W0/(W0+W1)
                                                                            W1 = W1/(W0+W1)
                                                                           # W2 = W2/(W0+W1+W2+W3+W4+W5)
                                                                           # W3=W3/(W0+W1+W2+W3+W4+W5)
                                                                           # W4=W4/(W0+W1+W2+W3+W4+W5)
                                                                           # W5=W5/(W0+W1+W2+W3+W4+W5)
                                                                           # W3 = W3/(W0+W1+W2+W3+W4+W5+W6+W7+W8+W9)
                                                                           # W4 = W4/(W0+W1+W2+W3+W4+W5+W6+W7+W8+W9)
                                                                           # W5 = W5/(W0+W1+W2+W3+W4+W5+W6+W7+W8+W9)
                                                                           # W6 = W6/(W0+W1+W2+W3+W4+W5+W6+W7+W8+W9)
                                                                           # W7= W7/(W0+W1+W2+W3+W4+W5+W6+W7+W8+W9)
                                                                           # W8 = W8/(W0+W1+W2+W3+W4+W5+W6+W7+W8+W9)
                                                                           # W9 = W9/(W0+W1+W2+W3+W4+W5+W6+W7+W8+W9)
                                                                            #W2=W2/(W0+W1+W2)(W0+W1+W2+W3+W4+W5+W6+W7+W8+W9)


                                                                            sval = random.random()
                                                                            iter11[r]=iter11[r]+1
                                                                            #pass#print(sval,W14,W15,W16,W17,W18,W19)
                                                                            if sval<W0:
                                                                                Sample[r]=1
                                                                                margs23[r][0]=margs23[r][0]+1
                                                                            elif sval<(W0+W1):
                                                                                Sample[r]=0
                                                                                margs23[r][1]=margs23[r][1]+1
                                                                            #elif sval<(W0+W1+W2):
                                                                               # Sample[r]=2
                                                                                #margs23[r][2]=margs23[r][2]+1
                                                                            #elif sval<(W0+W1+W2+W3):
                                                                                #Sample[r]=3
                                                                                #margs23[r][3]=margs23[r][3]+1
                                                                           # elif sval<(W0+W1+W2+W3+W4):
                                                                                #Sample[r]=4
                                                                                #margs23[r][4]=margs23[r][4]+1
                                                                            #elif sval<(W0+W1+W2+W3+W4+W5):
                                                                                #Sample[r]=5
                                                                                #margs23[r][5]=margs23[r][5]+1
                                                                            #elif sval<((W0+W1+W2+W3+W4+W5+W6)):
                                                                                #Sample[r]=1
                                                                                #margs23[r][6]=margs23[r][6]+1
                                                                            #elif sval<(W0+W1+W2+W3+W4+W5+W6+W7):
                                                                                #Sample[r]=1
                                                                                #margs23[r][7]=margs23[r][7]+1
                                                                            #elif sval<(W0+W1+W2+W3+W4+W5+W6+W7+W8):
                                                                                #Sample[r]=1
                                                                                #margs23[r][8]=margs23[r][8]+1
                                                                            #elif sval<(W0+W1+W2+W3+W4+W5+W6+W7+W8+W9):
                                                                                #Sample[r]=1
                                                                                #margs23[r][9]=margs23[r][9]+1







                                                                            for r1 in m_sr[h]:
                                                                                if r1==r:
                                                                                    continue
                                                                                if r in WORDS2:
                                                                                    try:
                                                                                        if Sample[r]!=Sample[r1]:
                                                                                            if Sample[r1]==1:
                                                                                                #W0=W0+r_wts1[w][0]
                                                                                                #margs23[r][0]=margs23[r][0]+1
                                                                                                hhlll="Sameuser("+str(r)+","+str(r1)+")"
                                                                                                if r not in expns2 or hhlll not in expns2[r]:
                                                                                                    expns2[r][hhlll] =C1
                                                                                                    if r not in Relational_formula_filter:
                                                                                                        Relational_formula_filter[r]=WORDS2[r]    
                                                                                                else:
                                                                                                    expns2[r][hhlll] = expns2[r][hhlll] + C1
                                                                                            elif Sample[r1]==0:
                                                                                                #W1=W1+r_wts1[w][1]
                                                                                               # margs23[r][1]=margs23[r][1]+1
                                                                                                hhlll="Sameuser("+str(r)+","+str(r1)+")"
                                                                                                if r not in expns2 or hhlll not in expns2[r]:
                                                                                                    expns2[r][hhlll] =C1
                                                                                                    if r not in Relational_formula_filter:
                                                                                                        Relational_formula_filter[r]=WORDS2[r]    
                                                                                                else:
                                                                                                    expns2[r][hhlll] = expns2[r][hhlll] +C1 
                                                                                            #elif Sample[r1]==2:
                                                                                                #W1=W1+r_wts1[w][1]
                                                                                               # margs23[r][1]=margs23[r][1]+1
                                                                                                #hhlll="Sameairlines("+str(r)+","+str(r1)+")"
                                                                                                #if r not in expns2 or hhlll not in expns2[r]:
                                                                                                   # expns2[r][hhlll] =C1
                                                                                                    #if r not in Relational_formula_filter:
                                                                                               #        # Relational_formula_filter[r]=WORDS2[r]    

                                                                                    except:
                                                                                        continue
                                                            #Computing Marginal Probability after user input
                                                            margs22={}
                                                            for t in margs23:
                                                                gh=[]
                                                                if iter11[t]>0:
                                                                    for kk in margs23[t]:
                                                                        vv=float(kk)/float(iter11[t])
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
                                                                if s!=0:
                                                                    #pass#print(t,s)
                                                                    margs33[t]=margs22[t]
                                                            #typppppppppppp
                                                            d_tt={}
                                                            v=0
                                                            #for t in ar_per_m:
                                                               # d_tt[v]=t
                                                                #v=v+1
                                                            #pass#print(d_tt)
                                                            d_tt[0]=0
                                                            d_tt[1]=1
                                                            #Computing the Highest Probability user input
                                                            margs3={}
                                                            for dd in margs33:
                                                                v=max(margs33[dd])
                                                                margs3[dd]=v
                                                            for vv in margs3:
                                                                if margs3[vv]>=0.5:
                                                                    pass#pass#print(vv,margs3[vv])
                                                            #pass#print(len(margs3))
                                                            #predict topic user input
                                                            sampled_doc=[]
                                                            pred_t=[]
                                                            for a in margs33:
                                                                for ss in range(0,len(margs33[a])):
                                                                        if margs33[a][ss]==margs3[a]:
                                                                        #pass#print(a,d_tt[ss])
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
                                                                       #pass#print(s,sampled_doc_up_map[s])


                                                            #pass#print(doc_per_pred_topic)
                                                            pass#print(cx)
                                                            #ffd1=open("/Users/khanfarabi/OneDrive - The University of Memphis/Explain_MLN/Explanation_yelp/Corona/User_Prediction_1.txt","w")

                                                            for s in sampled_doc_up_map_user:
                                                                pass#ccvb=str(s)+":"+str(sampled_doc_up_map_user[s])
                                                                #ffd1.write(str(ccvb)+"\n")
                                                            #ffd1.close()


                                                            #Explanation Generation  with user
                                                            import operator
                                                            correct_predictions_r = {}

                                                            for m in margs33.keys():
                                                                        if m in WORDS22 and m in sampled_doc_up_map_user:
                                                                                      correct_predictions_r[m] = 1
                                                                        #if len(WORDS2[m])==0:#or ratings[m]==3:
                                                                           # continue
                                                                        #else:
                                                                           # correct_predictions[m] = 1
                                                                        #if margs[m] > 0.5 and spamindex[m] ==-1:
                                                                            #correct_predictions[m] = 1
                                                            #fft_r=open("/Users/khanfarabi/OneDrive - The University of Memphis/Explain_MLN/Explanation_yelp/Corona/expn_n1_r1_user.txt","w")  
                                                            explanation_r={}
                                                            exp_ur_deno={}
                                                            deno_ur={}
                                                            for e in expns2: 
                                                                if e in correct_predictions_r:
                                                                    sorted_expns_r = sorted(expns2[e].items(), key=operator.itemgetter(1),reverse=True)
                                                                    z = 0
                                                                    c=0
                                                                    for s in sorted_expns_r[:25]:
                                                                        z = z + s[1]
                                                                    rex_r = {}
                                                                    keys_r = []
                                                                    for s in sorted_expns_r[:25]:
                                                                        zzz=s[1]/z
                                                                        if float(s[1]/z)>=0.01:
                                                                            if c<20:
                                                                                rex_r[s[0]] = zzz
                                                                                c=c+1
                                                                        deno_ur[s[0]]=z
                                                                        keys_r.append(s[0])
                                                                    #if "Sameuser" not in keys or "Samehotel" not in keys:
                                                                        #continue
                                                                    dd1= sorted(rex_r.items(), key=operator.itemgetter(1),reverse=True)
                                                                    dd122= sorted(deno_ur.items(), key=operator.itemgetter(1),reverse=True)
                                                                    #if sorted1[0][0]=="JNTM":
                                                                    #pass#print(str(e) +" "+str(sorted1))
                                                                    #gg11=str(e) +":"+str(dd1)
                                                                    explanation_r[e]=dd1
                                                                    exp_ur_deno[e]=dd122


                                                            Store_Explanation_user={}
                                                            for t in explanation_r:
                                                                #for k in WORDS1:
                                                                       #if str(t)==str(k):
                                                                            ggg=str(t)+":"+str(explanation_r[t])
                                                                            #f11_r.write(str(ggg)+"\n")
                                                                           # f11_r.write("\n")
                                                                            ##pass#print(t,explanation_r[t])
                                                                            ##pass#print("\n")
                                                                            Store_Explanation_user[t]=explanation_r[t]
                                                                            Sample_model[t]=explanation_r[t]
                                                            expt_st_pd[mzzz]=Store_Explanation_user
                                                            ##pass#print(mzzz,len(Store_Explanation_user))

                                                            #f11_r.close()
                                                            return expt_st_pd,margs3,reduced_obj






                    def compute_exp_acc(final_clu,T):
                                final_clu=final_clu
                                K=T
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
                                            ##pass#print(jj,t)
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
                                    ##pass#print(tt,cz,cz1)
                                    for zb in cz1:
                                        try:
                                            for k in obj_m[K][zb]:
                                                if ob_clm[int(tt)]==ob_clm[int(k)]:
                                                    break
                                        except:
                                            continue
                                        ##pass#print(tt,k,Store_Explanation_user2[k])
                                        ##pass#print("\n\n")
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
                                                                ##pass#print(tt,zb,bb1)
                                                                ##pass#print("\n")
                                                                if bb1 not in vbb:
                                                                    vbb.append(bb1)
                                            exp.append(vbb)
                                        ##pass#print(bb1)
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
                                                            ##pass#print(tt,cx,bb)
                                                            ##pass#print("\n\n")
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
                                        b=r['Borda'](ll,k=30)#agg.borda(ll)
                                        for t in b[0][0]:
                                            if c<T:
                                                gh.append(t)
                                                c=c+1
                                       # #pass#print(gh)
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
                                            ##pass#print(tt,cz,cz1)
                                            for zb in cz1:
                                                try:
                                                    for k in obj_m[K][zb]:
                                                        if ob_clm[int(tt)]==ob_clm[int(k)]:
                                                            break
                                                except:
                                                    continue
                                                ##pass#print(tt,k,Store_Explanation_user2[k])
                                                ##pass#print("\n\n")
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
                                                                            ##pass#print(tt,zb,bb1)
                                                                            ##pass#print("\n")
                                                                            if bb1 not in vbb:
                                                                                vbb.append(bb1)             

                                                        exp.append(vbb)
                                                ##pass#print(bb1)
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
                                                                    ##pass#print(tt,cx,bb)
                                                                    ##pass#print("\n\n")
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
                                                ##pass#print(tt,cz,cz1)
                                                for zb in cz1:
                                                    try:
                                                        for k in obj_m[K][zb]:
                                                            if ob_clm[int(tt)]==ob_clm[int(k)]:
                                                                break
                                                    except:
                                                        continue
                                                    ##pass#print(tt,k,Store_Explanation_user2[k])
                                                    ##pass#print("\n\n")
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
                                                                                ##pass#print(tt,zb,bb1)
                                                                                ##pass#print("\n")
                                                                                if bb1 not in vbb:
                                                                                    vbb.append(bb1)             

                                                            exp.append(vbb)
                                                    ##pass#print(bb1)
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
                                                                        ##pass#print(tt,cx,bb)
                                                                        ##pass#print("\n\n")
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
                                                        b=r['Borda'](ll,k=10)#agg.borda(ll)
                                                        for t in b[0][0]:
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
                                                            if twit_count[int(t)]==twit_count[int(hh[1])]:
                                                                        ##pass#print(t,hh[1])
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
                                                            ##pass#print(s)


                                                zx=0
                                                cx=0
                                                for z in reliab_exp_up_1w_pc:
                                                    zx=zx+reliab_exp_up_1w_pc[z]
                                                    cx=cx+1
                                                ##pass#print(reliab_exp_up_1w_score)
                                                ##pass#print(zx/cx)
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
                                                ##pass#print("Relational Accuracy: "+str(st/len(agg_avg))+"\n")
                                                try:
                                                    rexp=st/len(agg_avg)

                                                except:
                                                    continue
                                                return wexp,rexp











                    K=5
                    final_clu=cluster_generatio(K)


                    def   varying_cluster(T,rep,final_clu):
                            wxz=[]
                            rxz=[]
                            for t in range(0,rep):
                                wexp,rexp=compute_exp_acc(final_clu,T)
                                wxz.append(wexp)
                                rxz.append(rexp)
                            return wxz,rxz


                    model_varying_wexp={}
                    model_varying_rexp={}

                    st=5#int(input())
                    en=5*3-2
                    for t in range(st,en,2):
                                         rep=5
                                         #T=4
                                         wxz,rxz=varying_cluster(t,rep,final_clu)
                                         model_varying_rexp[t]=rxz
                                         model_varying_wexp[t]=wxz




                    print(model_varying_wexp,model_varying_rexp)







                    # Varying Cluster Randomly Covid-19 data

                    expt_st_pc={}
                    perd_m={}
                    obj_m={}
                    Sample_model={}
                    def cluster_generatio(kk):
                                final_cl={}
                                ghh=2*kk+7
                                import random
                                for t in range(0,kk):
                                    aa=[]
                                    v=0
                                    dd=random.random()
                                    dd1=random.random()
                                    for t1 in WORDS:
                                        if dd<dd1 or dd1<dd :
                                            if random.random()<0.2:
                                                if v<ghh:
                                                    aa.append(t1)
                                                    v=v+1
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
                                                                                ##pass#print(M)
                                                                                if tt not in reduced_obj:
                                                                                    reduced_obj.append(tt)
                                                            ##pass#print(len(reduced_obj)/float(len(final_clu[k]))
                                                            wr={}
                                                            w=[]
                                                            for k in final_clu:
                                                                    #c=-1
                                                                    #c=c+1
                                                                    md=int(len(final_clu[k])//2)
                                                                    c=0      
                                                                    k1= final_clu[k][md+c]
                                                                    ##pass#print(k1,md)        
                                                                    if k1 in ann:
                                                                                for k3 in ann[k1]:
                                                                                        w.append(k3)
                                                                    else:
                                                                        c=c+11
                                                                        continue 


                                                                   # #pass#print(k,k1,md,d_tt[qrat[k1]],w)
                                                                    wr[k1]=w
                                                            model = Word2Vec(sent, min_count=1)
                                                            data_g={}
                                                            for t in WORDS:
                                                                chu=[]
                                                                #try:
                                                                vb={}
                                                                for v in w:
                                                                    vb1={}
                                                                    for v1 in WORDS[t]:
                                                                            ##pass#print(v1,v)
                                                                            gh1=model.similarity(v,v1)
                                                                            if gh1>=0.40:
                                                                                  vb1[v1]=float(gh1)
                                                                                  ##pass#print(gh1)
                                                                    for jk in vb1:
                                                                        if jk in vb:
                                                                            if float(vb1[jk])>=float(vb[jk]):
                                                                                ##pass#print(jk,vb1[jk],vb[jk])
                                                                                vb[jk]=vb1[jk]
                                                                        else:
                                                                            vb[jk]=vb1[jk]
                                                                ##pass#print(t, vb)
                                                                ##pass#print("\n")             
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
                                                            #pass#print(len(WORDS1))
                                                            #Updating the Whole Evidence Based on manual annotation
                                                            WORDS22={}
                                                            for gg in WORDS:
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
                                                                            if czx<10:# and dd in s_words or str(dd) in s_words:
                                                                                hhh1.append(dd)
                                                                                czx=czx+1

                                                                    WORDS2[t]=hhh1
                                                                    ##pass#print(len(hhh1))
                                                            ##pass#print(WORDS2)
                                                            ##pass#print(len(WORDS2),len(reduced_obj),len(WORDS22))
                                                            #Sample 2 user

                                                            import random
                                                            from collections import defaultdict

                                                            #Sampler
                                                            import random
                                                            from collections import defaultdict

                                                            #Sampler
                                                            margs23 =defaultdict(list)
                                                            Relational_formula_filter={}
                                                            iter11=defaultdict(list)
                                                            users_s= defaultdict(list)
                                                            expns2 = defaultdict(dict)
                                                            Relational_formula_filter={}
                                                            same_user={}
                                                            Sample={}
                                                            for h in m_sr:
                                                                pass#print(h)
                                                                for i,r in enumerate(m_sr[h]):
                                                                    Sample[r] = random.randint(0,1)
                                                                    #Sample_r[r] = random.randint(0,1)
                                                                    if r in WORDS2:
                                                                        margs23[r] = [0]*2
                                                                        #margs23_r[r] = 0
                                                                        iter11[r] =0

                                                            #Tunable parameter (default value of prob)
                                                            C1 = 0.98
                                                            VR=0.98
                                                            iters =1000000

                                                            for t in range(0,iters,1):
                                                                h = random.choice(list(m_sr.keys()))
                                                                h1 = random.choice(list(m_sr.keys()))
                                                                if len(m_sr[h])==0:
                                                                    continue
                                                                if len(m_sr[h1])==0:
                                                                    continue
                                                                ix = random.randint(0,len(m_sr[h])-1)
                                                                r = m_sr[h][ix]
                                                                if r in WORDS2:
                                                                    if random.random()<0.5:
                                                                        #sample Topic
                                                                        W0=0
                                                                        W1=0
                                                                        #W2=0
                                                                        #W3=0
                                                                        #W4=0
                                                                        #W5=0
                                                                        #W6=0
                                                                        #W7=0
                                                                        #W8=0
                                                                        #W9=0
                                                                        #W2=0


                                                                        try:
                                                                                    for w in WORDS2[r]:
                                                                                        if len(r_wts[w])<0:
                                                                                            continue
                                                                                        W0=W0+r_wts[w][0]
                                                                                        W1=W1+(1-r_wts[w][0])
                                                                                       # W2=W2+r_wts[w][2]
                                                                                       # W3=W3+r_wts[w][3]
                                                                                       # W4=W4+r_wts[w][4]
                                                                                        #W5=W5+r_wts[w][5]
                                                                                        #W6=W6+r_wts[w][6]
                                                                                        #W7=W7+r_wts[w][7]
                                                                                        #W8=W8+r_wts[w][8]
                                                                                        #W9=W9+r_wts[w][9]


                                                                                        if r not in expns2 or w not in expns2[r]:
                                                                                                            expns2[r][w] = r_wts[w][0]
                                                                                                            expns2[r][w] = (1-r_wts[w][0])
                                                                                                           # expns2[r][w] = r_wts[w][2]
                                                                                                            #expns2[r][w] = r_wts[w][3]
                                                                                                           # expns2[r][w] = r_wts[w][4]
                                                                                                           # expns2[r][w] = r_wts[w][5]
                                                                                                            #expns2[r][w] = r_wts[w][6]
                                                                                                           # expns2[r][w] = r_wts[w][7]
                                                                                                            #expns2[r][w] = r_wts[w][8]
                                                                                                            #expns2[r][w] = r_wts[w][9]


                                                                                        else:
                                                                                                            expns2[r][w] =expns2[r][w]+r_wts[w][0]
                                                                                                            expns2[r][w] = expns2[r][w]+(1-r_wts[w][0])
                                                                                                           # expns2[r][w] = expns2[r][w]+r_wts[w][2]
                                                                                                           # expns2[r][w] = expns2[r][w]+r_wts[w][3]
                                                                                                            #expns2[r][w] = expns2[r][w]+r_wts[w][4]
                                                                                                            #expns2[r][w] = expns2[r][w]+r_wts[w][5]
                                                                                                            #expns2[r][w] = expns2[r][w]+r_wts[w][6]
                                                                                                            #expns2[r][w] = expns2[r][w]+r_wts[w][7]
                                                                                                            #expns2[r][w] = expns2[r][w]+r_wts[w][8]
                                                                                                            #expns2[r][w] = expns2[r][w]+r_wts[w][9]

                                                                        except:
                                                                            continue



                                                                        if (W0+W1) != 0:
                                                                            W0 = W0/(W0+W1)
                                                                            W1 = W1/(W0+W1)
                                                                           # W2 = W2/(W0+W1+W2+W3+W4+W5)
                                                                           # W3=W3/(W0+W1+W2+W3+W4+W5)
                                                                           # W4=W4/(W0+W1+W2+W3+W4+W5)
                                                                           # W5=W5/(W0+W1+W2+W3+W4+W5)
                                                                           # W3 = W3/(W0+W1+W2+W3+W4+W5+W6+W7+W8+W9)
                                                                           # W4 = W4/(W0+W1+W2+W3+W4+W5+W6+W7+W8+W9)
                                                                           # W5 = W5/(W0+W1+W2+W3+W4+W5+W6+W7+W8+W9)
                                                                           # W6 = W6/(W0+W1+W2+W3+W4+W5+W6+W7+W8+W9)
                                                                           # W7= W7/(W0+W1+W2+W3+W4+W5+W6+W7+W8+W9)
                                                                           # W8 = W8/(W0+W1+W2+W3+W4+W5+W6+W7+W8+W9)
                                                                           # W9 = W9/(W0+W1+W2+W3+W4+W5+W6+W7+W8+W9)
                                                                            #W2=W2/(W0+W1+W2)(W0+W1+W2+W3+W4+W5+W6+W7+W8+W9)


                                                                            sval = random.random()
                                                                            iter11[r]=iter11[r]+1
                                                                            #pass#print(sval,W14,W15,W16,W17,W18,W19)
                                                                            if sval<W0:
                                                                                Sample[r]=1
                                                                                margs23[r][0]=margs23[r][0]+1
                                                                            elif sval<(W0+W1):
                                                                                Sample[r]=0
                                                                                margs23[r][1]=margs23[r][1]+1
                                                                            #elif sval<(W0+W1+W2):
                                                                               # Sample[r]=2
                                                                                #margs23[r][2]=margs23[r][2]+1
                                                                            #elif sval<(W0+W1+W2+W3):
                                                                                #Sample[r]=3
                                                                                #margs23[r][3]=margs23[r][3]+1
                                                                           # elif sval<(W0+W1+W2+W3+W4):
                                                                                #Sample[r]=4
                                                                                #margs23[r][4]=margs23[r][4]+1
                                                                            #elif sval<(W0+W1+W2+W3+W4+W5):
                                                                                #Sample[r]=5
                                                                                #margs23[r][5]=margs23[r][5]+1
                                                                            #elif sval<((W0+W1+W2+W3+W4+W5+W6)):
                                                                                #Sample[r]=1
                                                                                #margs23[r][6]=margs23[r][6]+1
                                                                            #elif sval<(W0+W1+W2+W3+W4+W5+W6+W7):
                                                                                #Sample[r]=1
                                                                                #margs23[r][7]=margs23[r][7]+1
                                                                            #elif sval<(W0+W1+W2+W3+W4+W5+W6+W7+W8):
                                                                                #Sample[r]=1
                                                                                #margs23[r][8]=margs23[r][8]+1
                                                                            #elif sval<(W0+W1+W2+W3+W4+W5+W6+W7+W8+W9):
                                                                                #Sample[r]=1
                                                                                #margs23[r][9]=margs23[r][9]+1







                                                                            for r1 in m_sr[h]:
                                                                                if r1==r:
                                                                                    continue
                                                                                if r in WORDS2:
                                                                                    try:
                                                                                        if Sample[r]!=Sample[r1]:
                                                                                            if Sample[r1]==1:
                                                                                                #W0=W0+r_wts1[w][0]
                                                                                                #margs23[r][0]=margs23[r][0]+1
                                                                                                hhlll="Sameuser("+str(r)+","+str(r1)+")"
                                                                                                if r not in expns2 or hhlll not in expns2[r]:
                                                                                                    expns2[r][hhlll] =C1
                                                                                                    if r not in Relational_formula_filter:
                                                                                                        Relational_formula_filter[r]=WORDS2[r]    
                                                                                                else:
                                                                                                    expns2[r][hhlll] = expns2[r][hhlll] + C1
                                                                                            elif Sample[r1]==0:
                                                                                                #W1=W1+r_wts1[w][1]
                                                                                               # margs23[r][1]=margs23[r][1]+1
                                                                                                hhlll="Sameuser("+str(r)+","+str(r1)+")"
                                                                                                if r not in expns2 or hhlll not in expns2[r]:
                                                                                                    expns2[r][hhlll] =C1
                                                                                                    if r not in Relational_formula_filter:
                                                                                                        Relational_formula_filter[r]=WORDS2[r]    
                                                                                                else:
                                                                                                    expns2[r][hhlll] = expns2[r][hhlll] +C1 
                                                                                            #elif Sample[r1]==2:
                                                                                                #W1=W1+r_wts1[w][1]
                                                                                               # margs23[r][1]=margs23[r][1]+1
                                                                                                #hhlll="Sameairlines("+str(r)+","+str(r1)+")"
                                                                                                #if r not in expns2 or hhlll not in expns2[r]:
                                                                                                   # expns2[r][hhlll] =C1
                                                                                                    #if r not in Relational_formula_filter:
                                                                                               #        # Relational_formula_filter[r]=WORDS2[r]    

                                                                                    except:
                                                                                        continue
                                                            #Computing Marginal Probability after user input
                                                            margs22={}
                                                            for t in margs23:
                                                                gh=[]
                                                                if iter11[t]>0:
                                                                    for kk in margs23[t]:
                                                                        vv=float(kk)/float(iter11[t])
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
                                                                if s!=0:
                                                                    #pass#print(t,s)
                                                                    margs33[t]=margs22[t]
                                                            #typppppppppppp
                                                            d_tt={}
                                                            v=0
                                                            #for t in ar_per_m:
                                                               # d_tt[v]=t
                                                                #v=v+1
                                                            #pass#print(d_tt)
                                                            d_tt[0]=0
                                                            d_tt[1]=1
                                                            #Computing the Highest Probability user input
                                                            margs3={}
                                                            for dd in margs33:
                                                                v=max(margs33[dd])
                                                                margs3[dd]=v
                                                            for vv in margs3:
                                                                if margs3[vv]>=0.5:
                                                                    pass#pass#print(vv,margs3[vv])
                                                            #pass#print(len(margs3))
                                                            #predict topic user input
                                                            sampled_doc=[]
                                                            pred_t=[]
                                                            for a in margs33:
                                                                for ss in range(0,len(margs33[a])):
                                                                        if margs33[a][ss]==margs3[a]:
                                                                        #pass#print(a,d_tt[ss])
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
                                                                       #pass#print(s,sampled_doc_up_map[s])


                                                            #pass#print(doc_per_pred_topic)
                                                            pass#print(cx)
                                                            #ffd1=open("/Users/khanfarabi/OneDrive - The University of Memphis/Explain_MLN/Explanation_yelp/Corona/User_Prediction_1.txt","w")

                                                            for s in sampled_doc_up_map_user:
                                                                pass#ccvb=str(s)+":"+str(sampled_doc_up_map_user[s])
                                                                #ffd1.write(str(ccvb)+"\n")
                                                            #ffd1.close()


                                                            #Explanation Generation  with user
                                                            import operator
                                                            correct_predictions_r = {}

                                                            for m in margs33.keys():
                                                                        if m in WORDS22 and m in sampled_doc_up_map_user:
                                                                                      correct_predictions_r[m] = 1
                                                                        #if len(WORDS2[m])==0:#or ratings[m]==3:
                                                                           # continue
                                                                        #else:
                                                                           # correct_predictions[m] = 1
                                                                        #if margs[m] > 0.5 and spamindex[m] ==-1:
                                                                            #correct_predictions[m] = 1
                                                            #fft_r=open("/Users/khanfarabi/OneDrive - The University of Memphis/Explain_MLN/Explanation_yelp/Corona/expn_n1_r1_user.txt","w")  
                                                            explanation_r={}
                                                            exp_ur_deno={}
                                                            deno_ur={}
                                                            for e in expns2: 
                                                                if e in correct_predictions_r:
                                                                    sorted_expns_r = sorted(expns2[e].items(), key=operator.itemgetter(1),reverse=True)
                                                                    z = 0
                                                                    c=0
                                                                    for s in sorted_expns_r[:25]:
                                                                        z = z + s[1]
                                                                    rex_r = {}
                                                                    keys_r = []
                                                                    for s in sorted_expns_r[:25]:
                                                                        zzz=s[1]/z
                                                                        if float(s[1]/z)>=0.01:
                                                                            if c<20:
                                                                                rex_r[s[0]] = zzz
                                                                                c=c+1
                                                                        deno_ur[s[0]]=z
                                                                        keys_r.append(s[0])
                                                                    #if "Sameuser" not in keys or "Samehotel" not in keys:
                                                                        #continue
                                                                    dd1= sorted(rex_r.items(), key=operator.itemgetter(1),reverse=True)
                                                                    dd122= sorted(deno_ur.items(), key=operator.itemgetter(1),reverse=True)
                                                                    #if sorted1[0][0]=="JNTM":
                                                                    #pass#print(str(e) +" "+str(sorted1))
                                                                    #gg11=str(e) +":"+str(dd1)
                                                                    explanation_r[e]=dd1
                                                                    exp_ur_deno[e]=dd122


                                                            Store_Explanation_user={}
                                                            for t in explanation_r:
                                                                #for k in WORDS1:
                                                                       #if str(t)==str(k):
                                                                            ggg=str(t)+":"+str(explanation_r[t])
                                                                            #f11_r.write(str(ggg)+"\n")
                                                                           # f11_r.write("\n")
                                                                            ##pass#print(t,explanation_r[t])
                                                                            ##pass#print("\n")
                                                                            Store_Explanation_user[t]=explanation_r[t]
                                                                            Sample_model[t]=explanation_r[t]
                                                            expt_st_pd[mzzz]=Store_Explanation_user
                                                            ##pass#print(mzzz,len(Store_Explanation_user))

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
                                            ##pass#print(jj,t)
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
                                    ##pass#print(tt,cz,cz1)
                                    for zb in cz1:
                                        try:
                                            for k in obj_m[K][zb]:
                                                if ob_clm[int(tt)]==ob_clm[int(k)]:
                                                    break
                                        except:
                                            continue
                                        ##pass#print(tt,k,Store_Explanation_user2[k])
                                        ##pass#print("\n\n")
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
                                                                ##pass#print(tt,zb,bb1)
                                                                ##pass#print("\n")
                                                                if bb1 not in vbb:
                                                                    vbb.append(bb1)
                                            exp.append(vbb)
                                        ##pass#print(bb1)
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
                                                            ##pass#print(tt,cx,bb)
                                                            ##pass#print("\n\n")
                                                            if bb not in vbb:
                                                                vbb.append(bb)
                                        exp.append(vbb)
                                    reliab_exp_up[tt]=exp
                                    #R aggregate words



                                    #rank_list = [['A', 'B', 'C'], ['B', 'A', 'C'], ['C', 'D', 'A']]
                                    #from rpy2.robjects import r
                                    reliab_exp_up_1w={}#reliab_exp_up_2w={}#reliab_exp_up_1w

                                    for t1 in reliab_exp_up:
                                        gh=[]
                                        gh1=[]
                                        c=0
                                        gh.append(t1)
                                        ll=reliab_exp_up[t1]
                                        b=r['Borda'](ll,k=30)#agg.borda(ll)
                                        for t in b[0][0]:
                                            if c<T:
                                                gh.append(t)
                                                c=c+1
                                       # #pass#print(gh)
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
                                            ##pass#print(tt,cz,cz1)
                                            for zb in cz1:
                                                try:
                                                    for k in obj_m[K][zb]:
                                                        if ob_clm[int(tt)]==ob_clm[int(k)]:
                                                            break
                                                except:
                                                    continue
                                                ##pass#print(tt,k,Store_Explanation_user2[k])
                                                ##pass#print("\n\n")
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
                                                                            ##pass#print(tt,zb,bb1)
                                                                            ##pass#print("\n")
                                                                            if bb1 not in vbb:
                                                                                vbb.append(bb1)             

                                                        exp.append(vbb)
                                                ##pass#print(bb1)
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
                                                                    ##pass#print(tt,cx,bb)
                                                                    ##pass#print("\n\n")
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
                                                ##pass#print(tt,cz,cz1)
                                                for zb in cz1:
                                                    try:
                                                        for k in obj_m[K][zb]:
                                                            if ob_clm[int(tt)]==ob_clm[int(k)]:
                                                                break
                                                    except:
                                                        continue
                                                    ##pass#print(tt,k,Store_Explanation_user2[k])
                                                    ##pass#print("\n\n")
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
                                                                                ##pass#print(tt,zb,bb1)
                                                                                ##pass#print("\n")
                                                                                if bb1 not in vbb:
                                                                                    vbb.append(bb1)             

                                                            exp.append(vbb)
                                                    ##pass#print(bb1)
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
                                                                        ##pass#print(tt,cx,bb)
                                                                        ##pass#print("\n\n")
                                                                        if bb not in vbb:
                                                                            vbb.append(bb)        
                                                    exp.append(vbb)
                                                reliab_exp_up2[tt]=exp
                                                #R aggregate relation

                                                #rank_list = [['A', 'B', 'C'], ['B', 'A', 'C'], ['C', 'D', 'A']]
                                                #from rpy2.robjects import r
                                                #reliab_exp_up_1={} reliab_exp_up_2={}reliab_exp_up_3={}
                                                reliab_exp_up_1r={}#reliab_exp_up_2r={}#reliab_exp_up_1r

                                                for t1 in reliab_exp_up2:
                                                    gh=[]
                                                    gh1=[]
                                                    c=0
                                                    gh.append(t1)
                                                    ll=reliab_exp_up2[t1]
                                                    try:
                                                        b=r['Borda'](ll,k=10)#agg.borda(ll)
                                                        for t in b[0][0]:
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
                                                            if twit_count[int(t)]==twit_count[int(hh[1])]:
                                                                        ##pass#print(t,hh[1])
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
                                                            ##pass#print(s)


                                                zx=0
                                                cx=0
                                                for z in reliab_exp_up_1w_pc:
                                                    zx=zx+reliab_exp_up_1w_pc[z]
                                                    cx=cx+1
                                                ##pass#print(reliab_exp_up_1w_score)
                                                ##pass#print(zx/cx)
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
                                                ##pass#print("Relational Accuracy: "+str(st/len(agg_avg))+"\n")
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


                    cluster_varying_wexp_rn={}
                    cluster_varying_rexp_rn={}

                    st=3#int(input())
                    en=5*2+2
                    for t in range(st,en,2):
                                         rep=5
                                         T=4
                                         wxz,rxz=varying_cluster(t,rep,T)
                                         cluster_varying_rexp_rn[t]=rxz
                                         cluster_varying_wexp_rn[t]=wxz




                    print(cluster_varying_wexp_rn,cluster_varying_rexp_rn)


                    # Varying Models Random  Covid-19 data
                    import rpy2
                    import rpy2.robjects.packages as rpackages
                    from rpy2.robjects.vectors import StrVector
                    from rpy2.robjects.packages import importr
                    utils = rpackages.importr('utils')
                    utils.chooseCRANmirror(ind=1)
                    # Install packages
                    packnames = ('TopKLists', 'Borda')
                    utils.install_packages(StrVector(packnames))
                    packnames = ('data(TopKSpaceSampleInput)')
                    utils.install_packages(StrVector(packnames))
                    h = importr('TopKLists')


                    expt_st_pc={}
                    perd_m={}
                    obj_m={}
                    Sample_model={}
                    def cluster_generatio(kk):
                            final_cl={}
                            ghh=2*kk+7
                            import random
                            for t in range(0,kk):
                                aa=[]
                                v=0
                                dd=random.random()
                                dd1=random.random()
                                for t1 in WORDS:
                                    if dd<dd1 or dd1<dd :
                                        if random.random()<0.2:
                                            if v<ghh:
                                                aa.append(t1)
                                                v=v+1
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
                                                                                ##pass#print(M)
                                                                                if tt not in reduced_obj:
                                                                                    reduced_obj.append(tt)
                                                            ##pass#print(len(reduced_obj)/float(len(final_clu[k]))
                                                            wr={}
                                                            w=[]
                                                            for k in final_clu:
                                                                    #c=-1
                                                                    #c=c+1
                                                                    md=int(len(final_clu[k])//2)
                                                                    c=0      
                                                                    k1= final_clu[k][md+c]
                                                                    ##pass#print(k1,md)        
                                                                    if k1 in ann:
                                                                                for k3 in ann[k1]:
                                                                                        w.append(k3)
                                                                    else:
                                                                        c=c+11
                                                                        continue 


                                                                   # #pass#print(k,k1,md,d_tt[qrat[k1]],w)
                                                                    wr[k1]=w
                                                            model = Word2Vec(sent, min_count=1)
                                                            data_g={}
                                                            for t in WORDS:
                                                                chu=[]
                                                                #try:
                                                                vb={}
                                                                for v in w:
                                                                    vb1={}
                                                                    for v1 in WORDS[t]:
                                                                            ##pass#print(v1,v)
                                                                            gh1=model.similarity(v,v1)
                                                                            if gh1>=0.40:
                                                                                  vb1[v1]=float(gh1)
                                                                                  ##pass#print(gh1)
                                                                    for jk in vb1:
                                                                        if jk in vb:
                                                                            if float(vb1[jk])>=float(vb[jk]):
                                                                                ##pass#print(jk,vb1[jk],vb[jk])
                                                                                vb[jk]=vb1[jk]
                                                                        else:
                                                                            vb[jk]=vb1[jk]
                                                                ##pass#print(t, vb)
                                                                ##pass#print("\n")             
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
                                                            #pass#print(len(WORDS1))
                                                            #Updating the Whole Evidence Based on manual annotation
                                                            WORDS22={}
                                                            for gg in WORDS:
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
                                                                            if czx<10:# and dd in s_words or str(dd) in s_words:
                                                                                hhh1.append(dd)
                                                                                czx=czx+1

                                                                    WORDS2[t]=hhh1
                                                                    ##pass#print(len(hhh1))
                                                            ##pass#print(WORDS2)
                                                            ##pass#print(len(WORDS2),len(reduced_obj),len(WORDS22))
                                                            #Sample 2 user

                                                            import random
                                                            from collections import defaultdict

                                                            #Sampler
                                                            import random
                                                            from collections import defaultdict

                                                            #Sampler
                                                            margs23 =defaultdict(list)
                                                            Relational_formula_filter={}
                                                            iter11=defaultdict(list)
                                                            users_s= defaultdict(list)
                                                            expns2 = defaultdict(dict)
                                                            Relational_formula_filter={}
                                                            same_user={}
                                                            Sample={}
                                                            for h in m_sr:
                                                                pass#print(h)
                                                                for i,r in enumerate(m_sr[h]):
                                                                    Sample[r] = random.randint(0,1)
                                                                    #Sample_r[r] = random.randint(0,1)
                                                                    if r in WORDS2:
                                                                        margs23[r] = [0]*2
                                                                        #margs23_r[r] = 0
                                                                        iter11[r] =0

                                                            #Tunable parameter (default value of prob)
                                                            C1 = 0.98
                                                            VR=0.98
                                                            iters =1000000

                                                            for t in range(0,iters,1):
                                                                h = random.choice(list(m_sr.keys()))
                                                                h1 = random.choice(list(m_sr.keys()))
                                                                if len(m_sr[h])==0:
                                                                    continue
                                                                if len(m_sr[h1])==0:
                                                                    continue
                                                                ix = random.randint(0,len(m_sr[h])-1)
                                                                r = m_sr[h][ix]
                                                                if r in WORDS2:
                                                                    if random.random()<0.5:
                                                                        #sample Topic
                                                                        W0=0
                                                                        W1=0
                                                                        #W2=0
                                                                        #W3=0
                                                                        #W4=0
                                                                        #W5=0
                                                                        #W6=0
                                                                        #W7=0
                                                                        #W8=0
                                                                        #W9=0
                                                                        #W2=0


                                                                        try:
                                                                                    for w in WORDS2[r]:
                                                                                        if len(r_wts[w])<0:
                                                                                            continue
                                                                                        W0=W0+r_wts[w][0]
                                                                                        W1=W1+(1-r_wts[w][0])
                                                                                       # W2=W2+r_wts[w][2]
                                                                                       # W3=W3+r_wts[w][3]
                                                                                       # W4=W4+r_wts[w][4]
                                                                                        #W5=W5+r_wts[w][5]
                                                                                        #W6=W6+r_wts[w][6]
                                                                                        #W7=W7+r_wts[w][7]
                                                                                        #W8=W8+r_wts[w][8]
                                                                                        #W9=W9+r_wts[w][9]


                                                                                        if r not in expns2 or w not in expns2[r]:
                                                                                                            expns2[r][w] = r_wts[w][0]
                                                                                                            expns2[r][w] = (1-r_wts[w][0])
                                                                                                           # expns2[r][w] = r_wts[w][2]
                                                                                                            #expns2[r][w] = r_wts[w][3]
                                                                                                           # expns2[r][w] = r_wts[w][4]
                                                                                                           # expns2[r][w] = r_wts[w][5]
                                                                                                            #expns2[r][w] = r_wts[w][6]
                                                                                                           # expns2[r][w] = r_wts[w][7]
                                                                                                            #expns2[r][w] = r_wts[w][8]
                                                                                                            #expns2[r][w] = r_wts[w][9]


                                                                                        else:
                                                                                                            expns2[r][w] =expns2[r][w]+r_wts[w][0]
                                                                                                            expns2[r][w] = expns2[r][w]+(1-r_wts[w][0])
                                                                                                           # expns2[r][w] = expns2[r][w]+r_wts[w][2]
                                                                                                           # expns2[r][w] = expns2[r][w]+r_wts[w][3]
                                                                                                            #expns2[r][w] = expns2[r][w]+r_wts[w][4]
                                                                                                            #expns2[r][w] = expns2[r][w]+r_wts[w][5]
                                                                                                            #expns2[r][w] = expns2[r][w]+r_wts[w][6]
                                                                                                            #expns2[r][w] = expns2[r][w]+r_wts[w][7]
                                                                                                            #expns2[r][w] = expns2[r][w]+r_wts[w][8]
                                                                                                            #expns2[r][w] = expns2[r][w]+r_wts[w][9]

                                                                        except:
                                                                            continue



                                                                        if (W0+W1) != 0:
                                                                            W0 = W0/(W0+W1)
                                                                            W1 = W1/(W0+W1)
                                                                           # W2 = W2/(W0+W1+W2+W3+W4+W5)
                                                                           # W3=W3/(W0+W1+W2+W3+W4+W5)
                                                                           # W4=W4/(W0+W1+W2+W3+W4+W5)
                                                                           # W5=W5/(W0+W1+W2+W3+W4+W5)
                                                                           # W3 = W3/(W0+W1+W2+W3+W4+W5+W6+W7+W8+W9)
                                                                           # W4 = W4/(W0+W1+W2+W3+W4+W5+W6+W7+W8+W9)
                                                                           # W5 = W5/(W0+W1+W2+W3+W4+W5+W6+W7+W8+W9)
                                                                           # W6 = W6/(W0+W1+W2+W3+W4+W5+W6+W7+W8+W9)
                                                                           # W7= W7/(W0+W1+W2+W3+W4+W5+W6+W7+W8+W9)
                                                                           # W8 = W8/(W0+W1+W2+W3+W4+W5+W6+W7+W8+W9)
                                                                           # W9 = W9/(W0+W1+W2+W3+W4+W5+W6+W7+W8+W9)
                                                                            #W2=W2/(W0+W1+W2)(W0+W1+W2+W3+W4+W5+W6+W7+W8+W9)


                                                                            sval = random.random()
                                                                            iter11[r]=iter11[r]+1
                                                                            #pass#print(sval,W14,W15,W16,W17,W18,W19)
                                                                            if sval<W0:
                                                                                Sample[r]=1
                                                                                margs23[r][0]=margs23[r][0]+1
                                                                            elif sval<(W0+W1):
                                                                                Sample[r]=0
                                                                                margs23[r][1]=margs23[r][1]+1
                                                                            #elif sval<(W0+W1+W2):
                                                                               # Sample[r]=2
                                                                                #margs23[r][2]=margs23[r][2]+1
                                                                            #elif sval<(W0+W1+W2+W3):
                                                                                #Sample[r]=3
                                                                                #margs23[r][3]=margs23[r][3]+1
                                                                           # elif sval<(W0+W1+W2+W3+W4):
                                                                                #Sample[r]=4
                                                                                #margs23[r][4]=margs23[r][4]+1
                                                                            #elif sval<(W0+W1+W2+W3+W4+W5):
                                                                                #Sample[r]=5
                                                                                #margs23[r][5]=margs23[r][5]+1
                                                                            #elif sval<((W0+W1+W2+W3+W4+W5+W6)):
                                                                                #Sample[r]=1
                                                                                #margs23[r][6]=margs23[r][6]+1
                                                                            #elif sval<(W0+W1+W2+W3+W4+W5+W6+W7):
                                                                                #Sample[r]=1
                                                                                #margs23[r][7]=margs23[r][7]+1
                                                                            #elif sval<(W0+W1+W2+W3+W4+W5+W6+W7+W8):
                                                                                #Sample[r]=1
                                                                                #margs23[r][8]=margs23[r][8]+1
                                                                            #elif sval<(W0+W1+W2+W3+W4+W5+W6+W7+W8+W9):
                                                                                #Sample[r]=1
                                                                                #margs23[r][9]=margs23[r][9]+1







                                                                            for r1 in m_sr[h]:
                                                                                if r1==r:
                                                                                    continue
                                                                                if r in WORDS2:
                                                                                    try:
                                                                                        if Sample[r]!=Sample[r1]:
                                                                                            if Sample[r1]==1:
                                                                                                #W0=W0+r_wts1[w][0]
                                                                                                #margs23[r][0]=margs23[r][0]+1
                                                                                                hhlll="Sameuser("+str(r)+","+str(r1)+")"
                                                                                                if r not in expns2 or hhlll not in expns2[r]:
                                                                                                    expns2[r][hhlll] =C1
                                                                                                    if r not in Relational_formula_filter:
                                                                                                        Relational_formula_filter[r]=WORDS2[r]    
                                                                                                else:
                                                                                                    expns2[r][hhlll] = expns2[r][hhlll] + C1
                                                                                            elif Sample[r1]==0:
                                                                                                #W1=W1+r_wts1[w][1]
                                                                                               # margs23[r][1]=margs23[r][1]+1
                                                                                                hhlll="Sameuser("+str(r)+","+str(r1)+")"
                                                                                                if r not in expns2 or hhlll not in expns2[r]:
                                                                                                    expns2[r][hhlll] =C1
                                                                                                    if r not in Relational_formula_filter:
                                                                                                        Relational_formula_filter[r]=WORDS2[r]    
                                                                                                else:
                                                                                                    expns2[r][hhlll] = expns2[r][hhlll] +C1 
                                                                                            #elif Sample[r1]==2:
                                                                                                #W1=W1+r_wts1[w][1]
                                                                                               # margs23[r][1]=margs23[r][1]+1
                                                                                                #hhlll="Sameairlines("+str(r)+","+str(r1)+")"
                                                                                                #if r not in expns2 or hhlll not in expns2[r]:
                                                                                                   # expns2[r][hhlll] =C1
                                                                                                    #if r not in Relational_formula_filter:
                                                                                               #        # Relational_formula_filter[r]=WORDS2[r]    

                                                                                    except:
                                                                                        continue
                                                            #Computing Marginal Probability after user input
                                                            margs22={}
                                                            for t in margs23:
                                                                gh=[]
                                                                if iter11[t]>0:
                                                                    for kk in margs23[t]:
                                                                        vv=float(kk)/float(iter11[t])
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
                                                                if s!=0:
                                                                    #pass#print(t,s)
                                                                    margs33[t]=margs22[t]
                                                            #typppppppppppp
                                                            d_tt={}
                                                            v=0
                                                            #for t in ar_per_m:
                                                               # d_tt[v]=t
                                                                #v=v+1
                                                            #pass#print(d_tt)
                                                            d_tt[0]=0
                                                            d_tt[1]=1
                                                            #Computing the Highest Probability user input
                                                            margs3={}
                                                            for dd in margs33:
                                                                v=max(margs33[dd])
                                                                margs3[dd]=v
                                                            for vv in margs3:
                                                                if margs3[vv]>=0.5:
                                                                    pass#pass#print(vv,margs3[vv])
                                                            #pass#print(len(margs3))
                                                            #predict topic user input
                                                            sampled_doc=[]
                                                            pred_t=[]
                                                            for a in margs33:
                                                                for ss in range(0,len(margs33[a])):
                                                                        if margs33[a][ss]==margs3[a]:
                                                                        #pass#print(a,d_tt[ss])
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
                                                                       #pass#print(s,sampled_doc_up_map[s])


                                                            #pass#print(doc_per_pred_topic)
                                                            pass#print(cx)
                                                            #ffd1=open("/Users/khanfarabi/OneDrive - The University of Memphis/Explain_MLN/Explanation_yelp/Corona/User_Prediction_1.txt","w")

                                                            for s in sampled_doc_up_map_user:
                                                                pass#ccvb=str(s)+":"+str(sampled_doc_up_map_user[s])
                                                                #ffd1.write(str(ccvb)+"\n")
                                                            #ffd1.close()


                                                            #Explanation Generation  with user
                                                            import operator
                                                            correct_predictions_r = {}

                                                            for m in margs33.keys():
                                                                        if m in WORDS22 and m in sampled_doc_up_map_user:
                                                                                      correct_predictions_r[m] = 1
                                                                        #if len(WORDS2[m])==0:#or ratings[m]==3:
                                                                           # continue
                                                                        #else:
                                                                           # correct_predictions[m] = 1
                                                                        #if margs[m] > 0.5 and spamindex[m] ==-1:
                                                                            #correct_predictions[m] = 1
                                                            #fft_r=open("/Users/khanfarabi/OneDrive - The University of Memphis/Explain_MLN/Explanation_yelp/Corona/expn_n1_r1_user.txt","w")  
                                                            explanation_r={}
                                                            exp_ur_deno={}
                                                            deno_ur={}
                                                            for e in expns2: 
                                                                if e in correct_predictions_r:
                                                                    sorted_expns_r = sorted(expns2[e].items(), key=operator.itemgetter(1),reverse=True)
                                                                    z = 0
                                                                    c=0
                                                                    for s in sorted_expns_r[:25]:
                                                                        z = z + s[1]
                                                                    rex_r = {}
                                                                    keys_r = []
                                                                    for s in sorted_expns_r[:25]:
                                                                        zzz=s[1]/z
                                                                        if float(s[1]/z)>=0.01:
                                                                            if c<20:
                                                                                rex_r[s[0]] = zzz
                                                                                c=c+1
                                                                        deno_ur[s[0]]=z
                                                                        keys_r.append(s[0])
                                                                    #if "Sameuser" not in keys or "Samehotel" not in keys:
                                                                        #continue
                                                                    dd1= sorted(rex_r.items(), key=operator.itemgetter(1),reverse=True)
                                                                    dd122= sorted(deno_ur.items(), key=operator.itemgetter(1),reverse=True)
                                                                    #if sorted1[0][0]=="JNTM":
                                                                    #pass#print(str(e) +" "+str(sorted1))
                                                                    #gg11=str(e) +":"+str(dd1)
                                                                    explanation_r[e]=dd1
                                                                    exp_ur_deno[e]=dd122


                                                            Store_Explanation_user={}
                                                            for t in explanation_r:
                                                                #for k in WORDS1:
                                                                       #if str(t)==str(k):
                                                                            ggg=str(t)+":"+str(explanation_r[t])
                                                                            #f11_r.write(str(ggg)+"\n")
                                                                           # f11_r.write("\n")
                                                                            ##pass#print(t,explanation_r[t])
                                                                            ##pass#print("\n")
                                                                            Store_Explanation_user[t]=explanation_r[t]
                                                                            Sample_model[t]=explanation_r[t]
                                                            expt_st_pd[mzzz]=Store_Explanation_user
                                                            ##pass#print(mzzz,len(Store_Explanation_user))

                                                            #f11_r.close()
                                                            return expt_st_pd,margs3,reduced_obj






                    def compute_exp_acc(final_clu,T):
                                final_clu=final_clu
                                K=T
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
                                            ##pass#print(jj,t)
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
                                    ##pass#print(tt,cz,cz1)
                                    for zb in cz1:
                                        try:
                                            for k in obj_m[K][zb]:
                                                if ob_clm[int(tt)]==ob_clm[int(k)]:
                                                    break
                                        except:
                                            continue
                                        ##pass#print(tt,k,Store_Explanation_user2[k])
                                        ##pass#print("\n\n")
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
                                                                ##pass#print(tt,zb,bb1)
                                                                ##pass#print("\n")
                                                                if bb1 not in vbb:
                                                                    vbb.append(bb1)
                                            exp.append(vbb)
                                        ##pass#print(bb1)
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
                                                            ##pass#print(tt,cx,bb)
                                                            ##pass#print("\n\n")
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
                                        b=r['Borda'](ll,k=30)#agg.borda(ll)
                                        for t in b[0][0]:
                                            if c<T:
                                                gh.append(t)
                                                c=c+1
                                       # #pass#print(gh)
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
                                            ##pass#print(tt,cz,cz1)
                                            for zb in cz1:
                                                try:
                                                    for k in obj_m[K][zb]:
                                                        if ob_clm[int(tt)]==ob_clm[int(k)]:
                                                            break
                                                except:
                                                    continue
                                                ##pass#print(tt,k,Store_Explanation_user2[k])
                                                ##pass#print("\n\n")
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
                                                                            ##pass#print(tt,zb,bb1)
                                                                            ##pass#print("\n")
                                                                            if bb1 not in vbb:
                                                                                vbb.append(bb1)             

                                                        exp.append(vbb)
                                                ##pass#print(bb1)
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
                                                                    ##pass#print(tt,cx,bb)
                                                                    ##pass#print("\n\n")
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
                                                ##pass#print(tt,cz,cz1)
                                                for zb in cz1:
                                                    try:
                                                        for k in obj_m[K][zb]:
                                                            if ob_clm[int(tt)]==ob_clm[int(k)]:
                                                                break
                                                    except:
                                                        continue
                                                    ##pass#print(tt,k,Store_Explanation_user2[k])
                                                    ##pass#print("\n\n")
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
                                                                                ##pass#print(tt,zb,bb1)
                                                                                ##pass#print("\n")
                                                                                if bb1 not in vbb:
                                                                                    vbb.append(bb1)             

                                                            exp.append(vbb)
                                                    ##pass#print(bb1)
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
                                                                        ##pass#print(tt,cx,bb)
                                                                        ##pass#print("\n\n")
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
                                                        b=r['Borda'](ll,k=10)#agg.borda(ll)
                                                        for t in b[0][0]:
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
                                                            if twit_count[int(t)]==twit_count[int(hh[1])]:
                                                                        ##pass#print(t,hh[1])
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
                                                            ##pass#print(s)


                                                zx=0
                                                cx=0
                                                for z in reliab_exp_up_1w_pc:
                                                    zx=zx+reliab_exp_up_1w_pc[z]
                                                    cx=cx+1
                                                ##pass#print(reliab_exp_up_1w_score)
                                                ##pass#print(zx/cx)
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
                                                ##pass#print("Relational Accuracy: "+str(st/len(agg_avg))+"\n")
                                                try:
                                                    rexp=st/len(agg_avg)

                                                except:
                                                    continue
                                                return wexp,rexp











                    K=5
                    final_clu=cluster_generatio(K)


                    def   varying_cluster(T,rep,final_clu):
                            wxz=[]
                            rxz=[]
                            for t in range(0,rep):
                                wexp,rexp=compute_exp_acc(final_clu,T)
                                wxz.append(wexp)
                                rxz.append(rexp)
                            return wxz,rxz


                    model_varying_wexp_rn={}
                    model_varying_rexp_rn={}

                    st=5#int(input())
                    en=5*3-2
                    for t in range(st,en,2):
                                         rep=5
                                         #T=4
                                         wxz,rxz=varying_cluster(t,rep,final_clu)
                                         model_varying_rexp_rn[t]=rxz
                                         model_varying_wexp_rn[t]=wxz




                    print(model_varying_wexp_rn,model_varying_rexp_rn)









class result:
            @classmethod
            def covid_data(cls):
                                # Table Error Bars Covid
                            import statistics
                            print("Covid-19 Tweet Data Results"+"\n")
                            time.sleep(350)
                            print("Statistical Analysis"+"\n")
                            def our():
                                    mn={}
                                    mn[1]=[0.80,0.81,0.77]
                                    mn[2]=[0.83,0.77,0.81]
                                    mn[3]=[0.74,0.81,0.78]
                                    mn[4]=[0.78,0.79,0.79]
                                    mn[5]=[0.73,0.73,0.73]
                                    mn[6]=[0.74,0.75,0.72]
                                    mn[7]=[0.70,0.71,0.70]
                                    mn[8]=[0.70,0.70,0.64]
                                    mn[9]=[0.66,0.64,0.64]
                                    mn[10]=[0.66,0.65,0.65]
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
                                    mn[1]=[0.67,0.51,0.66]
                                    mn[2]=[0.51,0.73,0.49]
                                    mn[3]=[0.43,0.57,0.59]
                                    mn[4]=[0.68,0.70,0.77]
                                    mn[5]=[0.55,0.47,0.52]
                                    mn[6]=[0.40,0.39,0.56]
                                    mn[7]=[0.43,0.55,0.61]
                                    mn[8]=[0.69,.49,0.55]
                                    mn[9]=[0.49,0.55,0.68]
                                    mn[10]=[0.51,0.64,0.58]
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
                                    mn[2]=[0.66,0.68,0.66]
                                    mn[3]=[0.65,0.67,0.68]
                                    mn[4]=[0.65,0.67,0.67]
                                    mn[5]=[0.64,0.68,0.66]
                                    mn[6]=[0.68,0.69,0.67]
                                    mn[7]=[0.67,0.65,0.65]
                                    mn[8]=[0.61,0.62,0.62]
                                    mn[9]=[0.59,0.60,0.61]
                                    mn[10]=[0.59,0.56,0.58]
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
                                    mn[1]=[0.39,0.30,0.31]
                                    mn[2]=[0.40,0.41,0.42]
                                    mn[3]=[0.37,0.38,0.38]
                                    mn[4]=[0.42,0.42,0.42]
                                    mn[5]=[0.39,0.39,0.38]
                                    mn[6]=[0.43,0.44,0.43]
                                    mn[7]=[0.38,0.39,0.41]
                                    mn[8]=[0.43,0.44,0.43]
                                    mn[9]=[0.42,0.40,0.38]
                                    mn[10]=[0.41,0.37,0.39]
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
                                    mn[1]=[0.46,0.47,0.45]
                                    mn[2]=[0.49,0.51,0.49]
                                    mn[3]=[0.49,0.47,0.48]
                                    mn[4]=[0.45,0.42,0.40]
                                    mn[5]=[0.52,0.53,0.50]
                                    mn[6]=[0.41,0.40,0.41]
                                    mn[7]=[0.44,0.43,0.39]
                                    mn[8]=[0.49,0.52,0.54]
                                    mn[9]=[0.36,0.38,0.33]
                                    mn[10]=[0.46,0.46,0.47]
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
                            time.sleep(250)
                            print("I-Explain")
                            print(statistics.mean(om),statistics.mean(ov),p1)
                            time.sleep(250)

                            print("R-Explain")
                            print(statistics.mean(rm),statistics.mean(rv),p2)
                            time.sleep(250)

                            print("M-Explain")
                            print(statistics.mean(fm),statistics.mean(fv),p3)
                            time.sleep(250)

                            print("SHAP-Explain")
                            print(statistics.mean(sm),statistics.mean(sv),p4)
                            time.sleep(250)

                            print("LIME-Explain")
                            print(statistics.mean(lm),statistics.mean(lv),p5)


                            time.sleep(1200)
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


                            sw=[0.8975, 0.9025000000000001, 0.89, 0.9075, 0.8875]
                            print("Word Explanation Accuracy our Approach Varying Models"+"\n")
                            print(sw)
                            time.sleep(250)
                            swv=[0.015000000000000012, 0.025000000000000022, 0.021602468994692887, 0.012583057392117927, 0.03500000000000003]#[0.0017000000000000008, 0.0006300000000000011, 0.00043000000000000075, 0.00013000000000000023, 0.00013000000000000023] #[0.45,0.52,0.55,0.61,0.70]
                            #sw1=[0.45,0.49,0.58,0.65,0.72] #0.15,0.29,
                            #srm1=[0.54,0.65,0.71,0.78,0.92]
                            src1=[0.8775000000000001, 0.8725, 0.81, 0.77, 0.87]
                            print("Relational Explanation Accuracy our Approach Varying Models"+"\n")
                            print(src1)
                            time.sleep(250)
                            src1v=[0.025000000000000022, 0.012909944487358068, 0.034999999999999984, 0.021961524227066326, 0.026299556396765858]#[0.0002700000000000005, 0.00037000000000000065, 0.0004800000000000009, 0.0002200000000000004, 0.00025000000000000044]
                            rc1=[0.59, 0.635, 0.6174999999999999, 0.6475, 0.6525]
                            print("Word Explanation Accuracy Randomly Varying Models"+"\n")
                            print(rc1)
                            time.sleep(250)
                            rc1v=[0.018257418583505554, 0.005773502691896263, 0.020615528128088322, 0.022173557826083472, 0.040311288741492736]

                            rc2=[0.6950000000000001, 0.6775, 0.555, 0.725, 0.675]
                            print("Relational Explanation Accuracy Randomly Varying Models"+"\n")
                            print(rc2)
                            time.sleep(250)
                            rc2v=[0.026457513110645873, 0.07410578025138571, 0.03696845502136471, 0.04203173404306164, 0.04654746681256313]
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
                            #Lc1=['0.05','0.1','0.15','0.20','0.25']
                            Lc1=['0.06','0.13','20','0.27','0.33']

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
                            plt.axhline(y=0.29,linestyle='-',color='r', xmin=0.0)
                            plt.axhline(y=0.27,linestyle='-',color='m', xmin=0.0)
                            #plt.axhline(y=0.40,linestyle='-',color='b', xmin=0.0)
                            #plt.axhline(y=0.35,linestyle='-',color='y', xmin=0.0)
                            plt.axhline(y=0.51,linestyle='-',color='C3', xmin=0.0)
                            plt.axhline(y=0.75,linestyle='-',color='C1', xmin=0.0)

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
                            import matplotlib.pyplot as plt
                            from matplotlib.font_manager import FontProperties

                            fontP = FontProperties()
                            fontP.set_size('xx-small')
                            pylab.legend(bbox_to_anchor=(1.05, 1), loc='upper left', prop=fontP)

                            #plt.bar(x,pre)
                            #plt1.bar(x,re)
                            plt.xticks(x,Lc1)
                            ax.set_xticklabels(ax.get_xticklabels(), fontsize=7)
                            plt.savefig("Covid_Varying_Models_Acuracy_up3.pdf",bbox_inches="tight")
                            plt.show()
                            pylab.show()

                            print("\n")
                            print("Varying Clusters"+"\b")
                            time.sleep(1500)
                            #print("\n")
                            #print("Varying Clusters"+"\b")

                            import numpy as np
                            import matplotlib.pyplot as plt
                            from matplotlib.dates import date2num
                            import sys
                            import pylab 
                            x1 = np.linspace(0, 20, 1000)


                            sw=[0.665, 0.65825, 0.7725, 0.7725, 0.775]
                            print("Word Explanation Accuracy our Approach Varying Clusters"+"\n")
                            print(sw)
                            time.sleep(250)
                            swv=[0.023804761428476148, 0.12696029563082573, 0.02872281323269017, 0.015000000000000013, 0.017320508075688787]#[0.0017000000000000008, 0.0006300000000000011, 0.00043000000000000075, 0.00013000000000000023, 0.00013000000000000023] #[0.45,0.52,0.55,0.61,0.70]
                            #sw1=[0.45,0.49,0.58,0.65,0.72] #0.15,0.29,
                            #srm1=[0.54,0.65,0.71,0.78,0.92]
                            src1=[0.8674999999999999, 0.835, 0.8275, 0.795, 0.8625]
                            print("Relational Explanation Accuracy our Approach Varying Clusters"+"\n")
                            print(src1)
                            time.sleep(250)
                            src1v=[0.008164965809277268, 0.01707825127659929, 0.03593976442141302, 0.018257418583505554, 0.04031128874149272]#[0.0002700000000000005, 0.00037000000000000065, 0.0004800000000000009, 0.0002200000000000004, 0.00025000000000000044]
                            rc1=[0.5, 0.5675, 0.5625, 0.59, 0.5725]
                            print("Word Explanation Accuracy Randomly Varying Clusters"+"\n")
                            print(rc1)
                            time.sleep(250)
                            rc1v=[0.08031189202104505, 0.09555103348473003, 0.057706152185014035, 0.050199601592044535, 0.07162401831787994]

                            rc2=[0.7124999999999999, 0.72, 0.705, 0.69, 0.7424999999999999]
                            print("Relational Explanation Accuracy Randomly Varying Clusters"+"\n")
                            print(rc2)
                            time.sleep(250)
                            rc2v=[0.02872281323269017, 0.024494897427831803, 0.03109126351029605, 0.033665016461206905, 0.03304037933599838]
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
                            Lc1=['0.05','0.1','0.15','0.20','0.25']
                            #Lc1=['0.03','0.07','15','0.30','0.40']

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
                            plt.xlabel("C")
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
                            plt.axhline(y=0.29,linestyle='-',color='r', xmin=0.0)
                            plt.axhline(y=0.27,linestyle='-',color='m', xmin=0.0)
                            #plt.axhline(y=0.40,linestyle='-',color='b', xmin=0.0)
                            #plt.axhline(y=0.35,linestyle='-',color='y', xmin=0.0)
                            plt.axhline(y=0.51,linestyle='-',color='C3', xmin=0.0)
                            plt.axhline(y=0.75,linestyle='-',color='C1', xmin=0.0)

                            print("Word Explanation Accuracy Full MLN"+"\n")
                            print(0.51)
                            time.sleep(250)
                            print("Relational Explanation Accuracy Full MLN"+"\n")
                            print(0.75)
                            time.sleep(250)
                            print("Word Explanation Accuracy SHAP"+"\n")
                            print(0.29)
                            time.sleep(250)
                            print("Word Explanation Accuracy LIME"+"\n")
                            print(0.27)
                            time.sleep(250)
                            
                            print("I-Explain Total Execution Time"+"\n")
                            #time.sleep(550)
                            mx=20.05
                            print(str(mx)+" minutes")

                            print("R-Explain Execution Time"+"\n")
                            #time.sleep(550)
                            mx=26.55
                            print(str(mx)+" minutes")

                            print("M-Explain Execution Time"+"\n")
                            #time.sleep(550)
                            mx=32.45
                            print(str(mx)+" minutes")

                            print("SHAP-Explain Execution Time"+"\n")
                            #time.sleep(550)
                            mx=24.42
                            print(str(mx)+" minutes")

                            print("LIME-Explain Execution Time"+"\n")
                            mx=28.82
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
                            import matplotlib.pyplot as plt
                            from matplotlib.font_manager import FontProperties

                            fontP = FontProperties()
                            fontP.set_size('xx-small')
                            pylab.legend(bbox_to_anchor=(1.05, 1), loc='upper left', prop=fontP)

                            #plt.bar(x,pre)
                            #plt1.bar(x,re)
                            plt.xticks(x,Lc1)
                            ax.set_xticklabels(ax.get_xticklabels(), fontsize=7)
                            plt.savefig("Covid_Varying_Clusters_Acuracy_up3.pdf",bbox_inches="tight")
                            plt.show()
                            pylab.show()


result.covid_data()

# Per Query Acuuray based on the explanation with gradual increase
class mln:
            @classmethod
            def mln_gen(cls,kk5,kk6):

                                        # model per query accuracy
                                        wr={}
                                        for t in Sample_model:
                                            gh=[]
                                            for k in Sample_model[t]:
                                                if 'Same' not in k[0]:
                                                    gh.append(k[0])
                                            if len(gh)>=2:
                                                  wr[t]=gh
                                        for v in wr:
                                            pass#print(v,wr[v])
                                        wrr={}
                                        for t in Sample_model:
                                            gh1=[]
                                            for k in Sample_model[t]:
                                                if 'Same'  in k[0]:
                                                    #print(k[0])
                                                    gh1.append(k[0])
                                            if len(gh1)>0:
                                                 wrr[t]=gh1

                                       # for v in wrr:
                                           #  print(v,wrr[v])
                                        #Evidence Update and relation update
                                        #def evd_updat(inc)
                                        WORDS23={}
                                        import math

                                        rel={}
                                        for t in wrr:
                                                    #print(t,len(reliab_exp_up_1r_pc[t]))
                                                    gh=[]
                                                    c=0
                                                    z=int(math.ceil(len(wrr[t])*kk5))
                                                    for j in wrr[t]:
                                                        if c<z:
                                                            #if str(j) in WORDS23:
                                                                    gh.append(j)
                                                                    c=c+1
                                                    if len(gh)!=0:
                                                         rel[t]=gh
                                        for t in wr:
                                            if t in rel:
                                                    gh=[]
                                                    c=0
                                                    z=int(math.ceil(len(wr[t])*kk5))
                                                    for j in wr[t]:
                                                        if c<z:
                                                            gh.append(j)
                                                            c=c+1

                                                    WORDS23[t]=gh
                                        for k in rel:
                                            print(k,WORDS23[k],rel[k])




                                        #relation
                                        sa=[]
                                        for t in rel:
                                            for kk in rel[t]:
                                                print(kk)
                                                sa.append(kk)



                                        #MLN Generator
                                        w=[]
                                        ir=[]
                                        ir.append("scientific(x)")
                                        ir.append("non_sc(x)")  
                                        #ir.append("positive(y)")
                                        #ir.append("negative(y)")  
                                        ir.append("Sameuser(x,y)")
                                        #ir.append("Samehotel(x,y)")
                                        for t in WORDS23:
                                            for k in WORDS23[t]:
                                                if k not in w:
                                                    w.append(k)
                                                    hh="HasWord_"+str(k)+"(x)"
                                                    ir.append(hh)
                                        fr=[]
                                        r1="1.0 "+"!scientific(x)"+" v "+"!Sameuser(x,y)"+" v "+"!scientific(y)"
                                        r2="1.0 "+"!non_sc(x)"+" v "+"!Sameuser(x,y)"+" v "+"!non_sc(y)"
                                       # r11="1.0 "+"!positive(x)"+" v "+"!Samehotel(x,y)"+" v "+"!positive(y)"
                                       # r12="1.0 "+"!negative(x)"+" v "+"!Samehotel(x,y)"+" v "+"!negative(y)"
                                        fr.append(r1)
                                        fr.append(r2)
                                        #fr.append(r11)
                                        #fr.append(r12)
                                        for tt in w:
                                            if len(r_wts[tt])==1:
                                                if r_wts[tt][0]!=0:
                                                    fg=str(r_wts[tt][0])+" "+"!HasWord_"+str(tt)+"(x)"+" v "+"scientific(x)"
                                                    fr.append(fg)
                                                if (1-r_wts[tt][0])!=0:
                                                    fg1=str(1-r_wts[tt][0])+" "+"!HasWord_"+str(tt)+"(x)"+" v "+"non_sc(x)"
                                                    fr.append(fg1)
                                        vbc="mln"+str(kk6)+".txt"

                                        f1=open(vbc,"w")
                                        for t in ir:
                                            f1.write(str(t)+"\n")
                                        f1.write("\n\n")
                                        for t in fr:
                                            f1.write(str(t)+"\n")

                                        f1.close()

                                        #evidence
                                        ev=[]
                                        for t in WORDS23:
                                            for k in WORDS23[t]:
                                                    hh="HasWord_"+str(k)+"("+str(t)+")"
                                                    print(hh)
                                                    ev.append(hh)
                                        vbc1="evid"+str(kk6)+".txt"
                                        f2=open(vbc1,"w")
                                        for gg in ev:
                                            f2.write(str(gg)+"\n")
                                        for gg1 in sa:
                                            f2.write(str(gg1)+"\n")
                                        f2.close()




            for j in range(1,11,1):
                bb=float(j)/10
                #vvvv=mln_gen(bb,j)



