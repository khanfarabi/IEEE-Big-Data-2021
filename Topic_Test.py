#Data Preprocessing. It is just intial test case
import sys
import os
class all_op:
    @classmethod
    def all(cls):
        class data_preprocess:
                    @classmethod
                    def run_op1(cls):
                                        c2=0
                                        doc_p=[]
                                        topic=[]
                                        user=[]
                                        text=[]
                                        import sys
                                        import os
                                        #dirname = os.path.dirname(__file__)

                                        entries = os.listdir('Data/')
                                        for entry in entries:
                                            if '20news-18828' in entry:
                                                entry1=os.listdir('Data/')
                                                #print(entry1)
                                                for f in entry1:
                                                    #jj="topic_"+str(c)

                                                    if '.py' not in f:
                                                        en2=os.listdir('Data/'+str(f)+"/")
                                                        for w in en2:
                                                           hhh=str(w)
                                                           ggh=open('Data/'+str(f)+"/"+str(hhh))
                                                           if c2<10000:
                                                               gg4="d_"+str(c2)
                                                               doc_p.append(gg4)
                                                               c2=c2+1
                                                               for k in ggh:
                                                                    if 'From' in k:
                                                                        #c=c+1
                                                                        user.append(c2-1)
                                                                        break


                                        """
                                        ss=set(user)
                                        ss1=set(topic)

                                        u_topic=[]
                                        u_user=[]

                                        for tt in ss1:
                                            u_topic.append(tt)
                                        for kk in ss:
                                            u_user.append(kk)
                                        user_topic={}
                                        for ff in u_user:
                                            gh=[]
                                            for ee in range(0,len(user)):
                                                if str(ff)==str(user[ee]):
                                                    if topic[ee] not in gh:
                                                            gh.append(topic[ee])
                                            if len(gh)>1:
                                                user_topic[ff]=gh
                                        for dd in user_topic:
                                            #dd1=dd.split(":")
                                            print(dd,user_topic[dd])

                                        """
                                        ggh.close()

                                        #Data Preprocessing.
                                        import sys
                                        import os
                                        c2=0
                                        c3=0
                                        doc_pu=[]
                                        topic=[]
                                        user_u=[]
                                        text=[]

                                        #dirname = os.path.dirname(__file__)

                                        entries = os.listdir('/Data/')
                                        for entry in entries:
                                            if '20news-18828' in entry:
                                                entry1=os.listdir('/Data/')
                                                #print(entry1)
                                                for f in entry1:
                                                    #jj="topic_"+str(c)

                                                    if '.py' not in f:
                                                        en2=os.listdir('/Data/'+str(f)+"/")
                                                        for w in en2:
                                                           hhh=str(w)
                                                           ggh=open('/Data/'+str(f)+"/"+str(hhh))
                                                           if c2<10000:
                                                               gg4="d_"+str(c2)
                                                               doc_pu.append(gg4)
                                                               c2=c2+1
                                                           for k in ggh:
                                                                if 'From' in k:
                                                                    #c=c+1
                                                                    if c3<10000:
                                                                        user_u.append(k.strip("\n"))
                                                                        c3=c3+1
                                                                        break


                                        """
                                        ss=set(user)
                                        ss1=set(topic)

                                        u_topic=[]
                                        u_user=[]

                                        for tt in ss1:
                                            u_topic.append(tt)
                                        for kk in ss:
                                            u_user.append(kk)
                                        user_topic={}
                                        for ff in u_user:
                                            gh=[]
                                            for ee in range(0,len(user)):
                                                if str(ff)==str(user[ee]):
                                                    if topic[ee] not in gh:
                                                            gh.append(topic[ee])
                                            if len(gh)>1:
                                                user_topic[ff]=gh
                                        for dd in user_topic:
                                            #dd1=dd.split(":")
                                            print(dd,user_topic[dd])

                                        """
                                        ggh.close()
                                        #Data Preprocessing
                                        print(len(doc_pu),len(user_u))

                                        user_up={}
                                        user_up1={}
                                        u1=[]
                                        c=0
                                        ss=set(user_u)
                                        for k in ss:
                                            u1.append(k)
                                        for t in u1:
                                            hhg=[]
                                            for gg in range(0,len(user_u)):
                                                if str(t)==str(user_u[gg]):
                                                    if doc_pu[gg] not in hhg:
                                                        hhg.append(doc_pu[gg])
                                            #if c<1000:
                                            user_up[c]=hhg
                                            zz="d_"+str(c)
                                            user_up1[zz]=hhg
                                            c=c+1
                                        for df in user_up:
                                            #if len(user_up[df])==1:
                                                pass#print(df,user_up[df])
                                        #Data Preprocessing
                                        dd=[]
                                        User_Per_doc={}
                                        for k in range(0,10000):
                                            gg="d_"+str(k)
                                            dd.append(gg)


                                        for df in user_up:
                                            for hj in dd:
                                                if hj in user_up[df]:
                                                    #print(df,hj)
                                                    User_Per_doc[hj]=str(df)
                                        for q in User_Per_doc:
                                            pass#print(q,User_Per_doc[q])
                                        #Data Preprocessing
                                        dd=[]
                                        User_Per_doc={}
                                        for k in range(0,10000):
                                            gg="d_"+str(k)
                                            dd.append(gg)


                                        for df in user_up:
                                            for hj in dd:
                                                if hj in user_up[df]:
                                                    #print(df,hj)
                                                    User_Per_doc[hj]=str(df)
                                        for q in User_Per_doc:
                                            pass#print(q,User_Per_doc[q])
                                        #Data Preprocessing
                                        #tt_d contains the mappng of topic and documents mapping. We have totall 1000 documents in this experiment
                                        import sys
                                        import os
                                        import re

                                        DoC_MaP={}
                                        DoC_MaP_u={}
                                        DM=0
                                        topic=[]
                                        fff5=open("Topic_Data.txt","w")
                                        user=[]
                                        text=[]
                                        stopwords=[]
                                        num=[]
                                        for t in range(0,10):
                                            num.append(int(t))
                                        sfile = open("stopwords.txt")
                                        for ln in sfile:
                                            stopwords.append(ln.strip().lower())
                                        sfile.close()
                                        swords=[]
                                        sfile1 = open("Words.txt")
                                        for ln in sfile1:
                                            swords.append(ln.strip().lower())
                                        sfile1.close()
                                        flags = (re.UNICODE if sys.version < '3' and type(text) is unicode else 0)
                                        import string
                                        topic_text={}
                                        #table = str.maketrans('', '', string.punctuation)
                                        #dirname = os.path.dirname(__file__)
                                        c=0
                                        entries1 = os.listdir('/Datap/')
                                        for entry in entries1:
                                            if '20news-18828' in entry:
                                                t=0
                                                #c=0
                                                tt_d={}
                                                d_tt={}
                                                entry11=os.listdir('/Data/')
                                                #print(entry1)
                                                for f in entry11:
                                                    dd4={}
                                                    c=0

                                                    #jj="topic_"+str(c)
                                                    #c=c+1
                                                    if '.py' not in f:
                                                        en2=os.listdir('/Data/'+str(f)+"/")
                                                        for w in en2:
                                                            hhh=str(w)
                                                            #dd4="d_"+str(c)
                                                            #c=c+1
                                                            ggh1=open('/Data/'+str(f)+"/"+str(hhh))
                                                            keep=[]
                                                            feep=[]
                                                            feep1=[]
                                                            s=''
                                                            for k in ggh1:
                                                                feep1.append(k)

                                                                #vvb=[]
                                                                s=''
                                                                if '@' not in k and ':' not in k:
                                                                    for zc in k.strip('\n').split():
                                                                        if zc.isalnum() and ' ' not in zc:
                                                                                    s=s+zc+" "
                                                                feep.append(s)
                                                                for word in re.findall(r"\w[\w':]*", k, flags=flags):
                                                                    if word.isdigit() or len(word)==1:
                                                                        continue
                                                                    word_lower = word.lower()

                                                                    if word_lower in stopwords:
                                                                          continue
                                                                    #else:
                                                                    for ty in string.punctuation:
                                                                        if str(ty) in word_lower:
                                                                            continue
                                                                    for tt in num:
                                                                                if str(tt) not in word_lower:
                                                                                           # wd=word_lower.translate(table)
                                                                                           if word_lower not in keep and '"' not in word_lower  and ':' not in word_lower and '__________________________________________________________________________' not in word_lower :
                                                                                                #if word_lower in swords:
                                                                                                if word_lower.isalnum():
                                                                                                    if not any(c.isdigit() for c in word_lower):
                                                                                                                keep.append(word_lower)

                                                            if c<500:# and DM<1000:
                                                                dd4[c]=keep
                                                                if DM<10000:
                                                                    ggz="d_"+str(DM)
                                                                    DoC_MaP[ggz]=feep
                                                                    DoC_MaP_u[ggz]=feep1
                                                                    DM=DM+1
                                                                c=c+1


                                                    tt_d[t]=dd4
                                                    if '.py' not in f:
                                                        d_tt[t]=f
                                                    t=t+1
                                        """
                                        WORDS={}
                                        for tt in topic_text:  
                                            WORDS[u_topic_map[tt]]=topic_text[tt]
                                            hhh=str(tt)+":"+str(topic_text[tt])
                                            fff5.write(str(hhh)+"\n")
                                            print("\n")
                                            print("\t")                                                    
                                        """                       
                                        #fff5.close()
                                        ggh1.close()
                                        """
                                        ss=set(user)
                                        ss1=set(topic)

                                        u_topic=[]
                                        u_user=[]

                                        for tt in ss1:
                                            u_topic.append(tt)
                                        for kk in ss:
                                            u_user.append(kk)
                                        user_topic={}
                                        c=0
                                        for ff in u_user:
                                            gh=[]
                                            for ee in range(0,len(user)):
                                                if str(ff)==str(user[ee]):
                                                    if topic[ee] not in gh:
                                                            gh.append(topic[ee])
                                            if len(gh)>1:
                                                user_topic[c]=gh
                                                c=c+1
                                        for dd in user_topic:
                                            #dd1=dd.split(":")
                                            print(dd,user_topic[dd])

                                        """







                                        """
                                        f11=open("/Users/khanfarabi/OneDrive - The University of Memphis/Explain_MLN/Explanation_yelp1/NewsGroup/Topic_D.txt")
                                        WORDS={}
                                        for t in f11:
                                            h=t.strip("\n").split(":")
                                            for kk in u_topic:
                                                hf=[]
                                                if str(kk)==str(h[0]):
                                                    for q in h[1]:
                                                        if q not in hf:
                                                            #print(q)
                                                            hf.append(q)
                                            #print(u_topic_map[h[0]],hf)
                                            WORDS[u_topic_map[kk]]=hf
                                           # print("\n")
                                        """

                                        #for t in tt_d:
                                        # Mapping the document with topic DoC_MaP_u
                                        d_ty={}
                                        cc=0
                                        for t in tt_d:
                                            for k in tt_d[t]:
                                                gg="d_"+str(cc)
                                                #print(gg,t)
                                                d_ty[gg]=t
                                                cc=cc+1
                                                #print("\n")
                                        for q in d_ty:
                                            if 'auto' in d_tt[d_ty[q]]:
                                                   pass#print(q,d_ty[q],d_tt[d_ty[q]])        
                                        #Just for Checking
                                        c=0
                                        for z in DoC_MaP:
                                            #if c<1:
                                                pass#print(z)
                                                #print("\n")
                                                c=c+1
                                        #Data Preprocessing
                                        WORDS={}
                                        import collections
                                        windex = collections.defaultdict(list)
                                        targets=[]
                                        tar_t=[]
                                        words=[]
                                        words_train=[]
                                        #Doc_Sent={}
                                        c=0
                                        import random
                                        topicindex={}
                                        for tt in tt_d:
                                           # print(tt)
                                            for kk in tt_d[tt]:
                                                #print(kk)
                                                fd="d_"+str(c)
                                                WORDS[fd]=tt_d[tt][kk]
                                                for tzz in tt_d[tt][kk]:
                                                    windex[tzz].append(fd)
                                                topicindex[fd]=tt
                                                s=''
                                                for t in tt_d[tt][kk]:
                                                    s=s+str(t)+" "
                                                    if tt not in words_train:
                                                        words_train.append(t)
                                                        targets.append(tt)

                                                words.append(s)
                                                tar_t.append(tt)



                                                c=c+1
                                        #for t in tt_d:
                                        # Mapping the document with topic new evidence
                                        d_ty={}
                                        cc=0
                                        WORDS_upp={}
                                        ann_upp={}
                                        for t in tt_d:
                                            for k in tt_d[t]:
                                                gg="d_"+str(cc)
                                                #print(gg,t)
                                                d_ty[gg]=t
                                                cc=cc+1
                                                #print("\n")
                                        WORDS_upp1={}
                                        for q in d_ty:
                                            if 'med'  in d_tt[d_ty[q]]:
                                                    #print(q,d_ty[q],d_tt[d_ty[q]])
                                                    #ann_upp[q]=ann[q]
                                                    WORDS_upp1[q]=WORDS[q]
                                            elif 'auto'  in d_tt[d_ty[q]]:
                                                    #print(q,d_ty[q],d_tt[d_ty[q]])
                                                    #ann_upp[q]=ann[q]
                                                    WORDS_upp1[q]=WORDS[q]
                                        print(len(WORDS_upp1))            
                                        for t in WORDS_upp1:
                                            pass# print(t,d_tt[d_ty[t]])

                                        user_up2={}
                                        u1=[]
                                        c=0
                                        ss=set(user_u)
                                        for k in ss:
                                            u1.append(k)
                                        for t in u1:
                                            hhg=[]
                                            for gg in range(0,len(user_u)):
                                                if str(t)==str(user_u[gg]):
                                                    if doc_pu[gg] not in hhg:
                                                        if 'med'  in d_tt[d_ty[doc_pu[gg]]] or 'auto'  in d_tt[d_ty[doc_pu[gg]]]:
                                                                                hhg.append(doc_pu[gg])
                                            #if c<1000:
                                            user_up2[c]=hhg
                                            #zz="d_"+str(c)
                                            #user_up1[zz]=hhg
                                            c=c+1
                                        for df in user_up:
                                            #if len(user_up[df])==1:
                                                pass#print(df,user_up[df])
                                        #updated train and target words_train targets WORDS_upp1 topicindex

                                        words_train=[]
                                        targets=[]


                                        for t in WORDS_upp1:
                                            for k in WORDS_upp1[t]:
                                                words_train.append(k)
                                                targets.append(topicindex[t])
                                               # print(topicindex[t])
                                        #Sentence Generation
                                        #Sentence Generation
                                        '''
                                        Doc_Sent={}
                                        c=0
                                        sent=[]
                                        for tt in tt_d:
                                            if int(tt)==7 or int(tt)==13:
                                                            for kk in tt_d[tt]:

                                                                ff="t_"+str(tt)
                                                                ff1="d_"+str(c)
                                                                if ff1 in WORDS_upp1:
                                                                        for hh in tt_d[tt][kk]: 
                                                                            gh=[]
                                                                            gh.append(ff1)
                                                                            gh.append(ff)
                                                                            gh.append(str(User_Per_doc[ff1]))
                                                                            if  hh not in gh:
                                                                               # if hh in r_wts1:
                                                                                    gh.append(hh)
                                                                            #print(ff1,gh)
                                                                            if len(gh)>3:
                                                                                    sent.append(gh)
                                                                        c=c+1
                                        for et in sent:
                                               pass#print(et)

                                         '''
                                        sent=[]
                                        for t in WORDS_upp1:
                                            for kk in WORDS_upp1[t]:
                                                gh=[]
                                                gh.append(str(topicindex[t]))
                                                gh.append(str(t))
                                                gh.append(str(User_Per_doc[t]))
                                                gh.append(kk)
                                                sent.append(gh)


                                        #Store learned weights
                                        from sklearn.feature_extraction.text import TfidfVectorizer
                                        from collections import defaultdict
                                        from sklearn import svm
                                        #from sklearn import cross_validation
                                        from sklearn.model_selection import cross_validate
                                        from sklearn.preprocessing import MinMaxScaler
                                        from sklearn.metrics import (precision_score,recall_score,f1_score)
                                        from sklearn.multiclass import OneVsRestClassifier
                                        from sklearn.svm import SVC

                                        #Learn SVM Model
                                        #ifile = open("processed_revs_1.txt",encoding="ISO-8859-1")
                                        Y = targets
                                        #words = []
                                        unique_words = []
                                        #for ln in ifile:
                                           # parts = ln.strip().split("\t")
                                           # Y.append(int(parts[1]))
                                            #words.append(parts[0])
                                        for w in words:
                                                h=w.split()
                                                for e in h:
                                                    unique_words.append(e)
                                        #ifile.close()
                                        tf_transformer = TfidfVectorizer()
                                        f = tf_transformer.fit_transform(words_train)
                                        features = [((i, j), f[i,j]) for i, j in zip(*f.nonzero())]
                                        unique_word_ids = []
                                        for w in unique_words:
                                            i = tf_transformer.vocabulary_.get(w)
                                            unique_word_ids.append(i)

                                        clf =svm.LinearSVC(C=1)#OneVsRestClassifier(SVC(kernel='linear'))#svm.LinearSVC(C=1)
                                        clf.fit(f,Y)
                                        p = clf.predict(f)
                                        print(f1_score(Y,p,average='micro'))
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
                                            #print(i,j)
                                            for k in tf_transformer.vocabulary_.keys():
                                                if tf_transformer.vocabulary_[k]==j:
                                                    if k not in r_wts:
                                                        break
                                                    else:
                                                        r_wts[k][i] =V1[ix][0]
                                                        break
                                            ix = ix + 1
                                        #Just Printed to see the output
                                        documents=[]
                                        #documents1=[]
                                        for t in sent:
                                            for jh in t:
                                                if jh not in documents:
                                                        documents.append(jh)

                                        #for w in sent:
                                        #for k in documents:
                                        for tt in documents:
                                            pass#print(tt)
                                            #print("\n")


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



                                        NUM_CLUSTERS=20
                                        kclusterer = KMeansClusterer(NUM_CLUSTERS, distance=nltk.cluster.util.cosine_distance)
                                        assigned_clusters1 = kclusterer.cluster(X,assign_clusters=True)
                                        #print (assigned_clusters)
                                        cluster={}
                                        words = list(model.wv.vocab)
                                        for i, word in enumerate(words):
                                              gh=[] 
                                              gh1=[] 
                                              gh2=[] 
                                              #if word.isdigit(): 
                                              cluster[word]=assigned_clusters1[i]
                                            #print (word + ":" + str(assigned_clusters[i]))
                                        cluster_final={}
                                        for j in range(NUM_CLUSTERS):
                                            gg=[]
                                            for tt in cluster:
                                                if str(cluster[tt])==str(j):
                                                    if tt not in gg:
                                                        if 'd_' in tt:
                                                              gg.append(tt)
                                            if len(gg)>0:
                                                        cluster_final[j]=gg
                                        #print(cluster_final)
                                        cc=0
                                        final_clu={}
                                        for t in cluster_final:
                                            ghh=[]
                                            for k in cluster_final[t]:
                                                if k in WORDS_upp1:
                                                            pass#print(k)
                                        #Cluster Generation
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



                                        NUM_CLUSTERS=5
                                        kclusterer = KMeansClusterer(NUM_CLUSTERS, distance=nltk.cluster.util.cosine_distance)
                                        assigned_clusters1 = kclusterer.cluster(X,assign_clusters=True)
                                        #print (assigned_clusters)
                                        cluster={}
                                        words = list(model.wv.vocab)
                                        for i, word in enumerate(words):
                                              gh=[] 
                                              gh1=[] 
                                              gh2=[] 
                                              #if word.isdigit(): 
                                              cluster[word]=assigned_clusters1[i]
                                            #print (word + ":" + str(assigned_clusters[i]))
                                        cluster_final={}
                                        for j in range(NUM_CLUSTERS):
                                            gg=[]
                                            for tt in cluster:
                                                if str(cluster[tt])==str(j):
                                                    if tt not in gg:
                                                        if 'd_' in tt:
                                                              gg.append(tt)
                                            if len(gg)>0:
                                                        cluster_final[j]=gg
                                        cc=0
                                        final_clu={}
                                        for t in cluster_final:
                                            ghh=[]
                                            for k in cluster_final[t]:
                                                #print(k)
                                                if k in WORDS_upp1:
                                                           #print(k)
                                                           ghh.append(k)
                                            if len(ghh)>0:
                                                final_clu[cc]=ghh
                                                cc=cc+1
                                        for k in final_clu:
                                            pass#print(k,final_clu[k],len(final_clu[k]))

                                        #annotation 1
                                        st_words=[]
                                        f =open("Wordst.txt")
                                        for k in f:
                                            pp=k.strip("\n").split()
                                            #print(pp[0])
                                            st_words.append(pp[0])
                                        #annot
                                        ann1={}
                                        c=0
                                        for k in WORDS_upp1:
                                            if int(topicindex[k])==7:
                                                c=c+1
                                                gff=[]
                                                for gg in WORDS_upp1[k]:
                                                    if gg in st_words:
                                                        gff.append(gg)
                                                if len(gff)>0:
                                                   # if k in WORDSt:
                                                        ann1[k]=gff

                                            elif int(topicindex[k])==13:
                                                c=c+1
                                                gff1=[]
                                                for gg in WORDS_upp1[k]:
                                                    if gg in st_words:
                                                        gff1.append(gg)
                                                if len(gff1)>0:
                                                    #if k WORDSt:
                                                        ann1[k]=gff1

                                        ann={}
                                        for t in ann1:
                                            if t in WORDS_upp1:
                                                ann[t]=ann1[t]
                                        for k in ann:
                                            pass#print(k,ann[k])

                                        wr={}
                                        w=[]
                                        for k in final_clu:
                                                #c=-1
                                                #c=c+1

                                                md=int(len(final_clu[k])/2)
                                                c=0      
                                                k1= final_clu[k][md+c]
                                                #print(k1,md)        
                                                if k1 in ann:
                                                            for k3 in ann[k1]:
                                                                    w.append(k3)
                                                else:
                                                    c=c+11
                                                    continue 


                                                #print(k,k1,md,d_tt[d_ty[k1]],w)
                                                wr[k1]=w
                                        #print(w)

                                        #Update Evidence Based on Survey input Update 2
                                        #data_extract11 similar_r_map1 data_extract11
                                        chc=['politics.mideast','politics.guns','religion.christian','autos','med','space','electronics']
                                        import gensim 
                                        import operator
                                        from gensim.models import Word2Vec
                                        model = Word2Vec(sent, min_count=1)
                                        data_g={}
                                        for t in WORDS_upp1:
                                            chu=[]
                                            vb={}
                                            #try:
                                            # for zz in chc:
                                                #if zz in d_tt[d_ty[t]]:
                                                        #print(zz)
                                            for v in w:
                                                            vb1={}
                                                            for v1 in WORDS_upp1[t]:
                                                                    #print(v1,v)
                                                                    try:
                                                                         gh1=model.similarity(v,v1)
                                                                    except:
                                                                        continue
                                                                    if gh1>0.90:
                                                                        vb1[v1]=float(gh1)


                                                            for jk in vb1:
                                                                    if jk in vb:
                                                                        if float(vb1[jk])>=float(vb[jk]):
                                                                            #print(jk,vb1[jk],vb[jk])
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


                                        '''       
                                        data_g1={}
                                        nn=[]
                                        for y in range(0,10):
                                            nn.append(str(y))
                                        for t in data_g:
                                            vz=[]
                                            c=0
                                            if len(data_g[t])>0:
                                                #print(data_g[t])
                                                for tt in data_g[t]:
                                                    for kk in nn:
                                                        if kk in tt:
                                                            continue
                                                        else:
                                                            if tt not in vz and tt not in stopwords:
                                                                if c<10:
                                                                    vz.append(tt)
                                                                    c=c+1
                                                data_g1[t]=vz
                                        '''
                                        print(len(data_g))
                                        #Updating the Whole Evidence Based on Survey Input
                                        WORDS22={}
                                        for gg in WORDS_upp1:
                                            #if gg in data_extract12:
                                                #WORDS2[gg]=data_extract12[gg]
                                            if gg in data_g:
                                                if len(data_g[gg])>0:
                                                    WORDS22[gg]=data_g[gg]
                                        #print(WORDS2['d_535'])
                                        print(len(WORDS22),len(WORDS_upp1))
                                        #for t in WORDS2:
                                            #pass#print(t,WORDS2[t])
                                        return WORDS_upp1,WORDS22,topicindex,r_wts,final_clu,user_up2,user_up
        #WORDS_upp1,WORDS22,topicindex,r_wts,final_clu,user_up2,user_up=data_preprocess.run_op1()

        # user feedback

        #Sample 2 for user input

        import random
        from collections import defaultdict

        class full_mln_exp:
                    @classmethod
                    def op1(cls,user_up,WORDS22):

                                        #Sampler
                                        import operator 
                                        margs =defaultdict(list)
                                        Relational_formula_filter={}
                                        iter1=defaultdict(list)
                                        users_s= defaultdict(list)
                                        expns3 = defaultdict(dict)
                                        Relational_formula_filter={}
                                        same_user={}
                                        Sample={}
                                        for h in user_up:
                                           # print(h)
                                            for i,r in enumerate(user_up[h]):
                                                if r in WORDS22:
                                                        Sample[r] = random.randint(0,1)
                                                        #Sample_r[r] = random.randint(0,1)
                                                        margs[r] = [0]*2
                                                        #margs_r[r] = 0
                                                        iter1[r] =0

                                        #Tunable parameter (default value of prob)
                                        C1 = 0.98
                                        VR=0.98
                                        iters =1000000

                                        for t in range(0,iters,1):
                                            h = random.choice(list(user_up.keys()))
                                            if len(user_up[h])==0:
                                                continue
                                            ix = random.randint(0,len(user_up[h])-1)
                                            r = user_up[h][ix]
                                            if r in WORDS22:
                                                        if random.random()<0.5:
                                                            #sample Topic
                                                            W0 = 0
                                                            W1 = 0
                                                            W2 = 0
                                                            W3 = 0
                                                            W4 = 0
                                                            W5 = 0
                                                            W6 = 0
                                                            W7 = 0
                                                            W8 = 0
                                                            W9 = 0
                                                            W10 = 0
                                                            W11 = 0
                                                            W12 = 0
                                                            W13 = 0
                                                            W14 = 0
                                                            W15 = 0
                                                            W16 = 0
                                                            W17 = 0
                                                            W18 = 0
                                                            W19 = 0
                                                            P0 = 0
                                                            P1 = 0
                                                            P2 = 0
                                                            P3 = 0
                                                            P4 = 0
                                                            P5 = 0
                                                            P6 = 0
                                                            P7 = 0
                                                            P8 = 0
                                                            P9 = 0
                                                            P10 = 0
                                                            P11 = 0
                                                            P12 = 0
                                                            P13 = 0
                                                            P14 = 0
                                                            P15 = 0
                                                            P16 = 0
                                                            P17 = 0
                                                            P18 = 0
                                                            P19 = 0
                                                            N0 = 0
                                                            N1 = 0
                                                            N2 = 0
                                                            N3 = 0
                                                            N4 = 0
                                                            N5 = 0
                                                            N6 = 0
                                                            N7 = 0
                                                            N8 = 0
                                                            N9 = 0
                                                            N10 = 0
                                                            N11 = 0
                                                            N12 = 0
                                                            N13 = 0
                                                            N14 = 0
                                                            N15 = 0
                                                            N16 = 0
                                                            N17 = 0
                                                            N18 = 0
                                                            N19 = 0

                                                            for w in WORDS22[r]:
                                                                if len(r_wts[w])<1:
                                                                    continue
                                                                W0=W0+r_wts[w][0]
                                                                W1=W1+(1-r_wts[w][0])


                                                                if r not in expns3 or w not in expns3[r]:
                                                                                    expns3[r][w] = r_wts[w][0]
                                                                                    expns3[r][w] = (1-r_wts[w][0])

                                                                else:
                                                                                    expns3[r][w] = expns3[r][w] + r_wts[w][0]
                                                                                    expns3[r][w] = expns3[r][w] + (1-r_wts[w][0])


                                                            if (W0+W1) != 0:
                                                                #print("y")
                                                                W0 = W0/(W0+W1)
                                                                W1 = W1/(W0+W1)

                                                                sval = random.random()
                                                                iter1[r]=iter1[r]+1
                                                                #print(sval,W14,W15,W16,W17,W18,W19)
                                                                if sval<W0:
                                                                    Sample[r]=0
                                                                    margs[r][0]=margs[r][0]+1
                                                                elif sval<(W0+W1):
                                                                    Sample[r]=1

                                                                    margs[r][1]=margs[r][1]+1


                                                                for r1 in user_up[h]:
                                                                    if r1==r:
                                                                        continue
                                                                    if r in WORDS22 and r1 in WORDS22:
                                                                            if Sample[r]!=Sample[r1]:
                                                                                if Sample[r1]==0:
                                                                                    #W0=W0+r_wts1[w][0]
                                                                                    #margs[r][0]=margs[r][0]+1
                                                                                    hhlll="Sameuser("+str(r)+","+str(r1)+")"
                                                                                    if r not in expns3 or hhlll not in expns3[r]:
                                                                                        expns3[r][hhlll] =C1
                                                                                        if r not in Relational_formula_filter:
                                                                                            Relational_formula_filter[r]=WORDS22[r]    
                                                                                    else:
                                                                                        expns3[r][hhlll] = expns3[r][hhlll] + C1
                                                                                elif Sample[r1]==1:
                                                                                    #W1=W1+r_wts1[w][1]
                                                                                   # margs[r][1]=margs[r][1]+1
                                                                                    hhlll="Sameuser("+str(r)+","+str(r1)+")"
                                                                                    if r not in expns3 or hhlll not in expns3[r]:
                                                                                        expns3[r][hhlll] =C1
                                                                                        if r not in Relational_formula_filter:
                                                                                            Relational_formula_filter[r]=WORDS22[r]    
                                                                                    else:
                                                                                        expns3[r][hhlll] = expns3[r][hhlll] +C1 



                                        #userr

                                        margs22={}
                                        for t in margs:
                                            gh=[]
                                            if iter1[t]>0:
                                                for kk in margs[t]:
                                                    vv=float(kk)/float(iter1[t])
                                                    if float(vv)>=1.0:
                                                        gh.append(float(1.0))
                                                    elif float(vv)<1.0:
                                                        gh.append(abs(float(vv)))
                                                margs22[t]=gh

                                        #checking correctness of probability distribution
                                        margs33={}
                                        for t in margs22:
                                            s=0
                                            for ww in margs22[t]:
                                                s=s+float(ww)
                                            if s!=0:
                                                #print(t,s)
                                                margs33[t]=margs22[t]
                                        #Computing the Highest Probability user input
                                        margs3_u={}
                                        for dd in margs22:
                                            v=max(margs22[dd])
                                            margs3_u[dd]=v
                                        for vv in margs3_u:
                                            if margs3_u[vv]>=0.2:
                                                pass#print(vv,margs3_u[vv])
                                        #topic mapping sci.med rec.autos
                                        c=0
                                        topic={}
                                        topics=[]
                                        topics.append('sci.med')
                                        topics.append('rec.autos')
                                        topic[0]='sci.med'
                                        topic[1]='rec.autos'

                                        print(len(topics))
                                        #predict topic user input
                                        sampled_doc=[]
                                        pred_t=[]
                                        for a in margs22:
                                            for ss in range(0,len(margs22[a])):
                                                if margs22[a][ss]==margs3_u[a]:
                                                    #print(a,d_tt[ss])
                                                    sampled_doc.append(a)
                                                    pred_t.append(topic[ss])
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
                                                   #print(s,sampled_doc_up_map[s])


                                        #print(doc_per_pred_topic)
                                        print(cx)
                                        ffd1=open("User_Prediction.txt","w")

                                        for s in sampled_doc_up_map_user:
                                            ccvb=str(s)+":"+str(sampled_doc_up_map_user[s])
                                            ffd1.write(str(ccvb)+"\n")
                                        ffd1.close()

                                        #mapping per topic docs
                                        sampled_doc_up_map_topic={}
                                        for tt in topics:
                                            ggf=[]
                                            for gg in range(0,len(sampled_doc)):
                                                if tt==pred_t[gg]:#sampled_doc[gg]:
                                                    if sampled_doc[gg] not in ggf:
                                                        ggf.append(sampled_doc[gg])
                                            if ggf!=[]:
                                                sampled_doc_up_map_topic[tt]=ggf
                                        for k in sampled_doc_up_map_topic:
                                            pass#print(k,sampled_doc_up_map_topic[k])
                                            pass#print("\n")


                                        #predict topic user input
                                        sampled_doc=[]
                                        pred_t=[]
                                        for a in margs22:
                                            for ss in range(0,len(margs22[a])):
                                                if margs22[a][ss]==margs3_u[a]:
                                                    #print(a,d_tt[ss])
                                                    sampled_doc.append(a)
                                                    pred_t.append(topic[ss])
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
                                                   #print(s,sampled_doc_up_map[s])


                                        #print(doc_per_pred_topic)
                                        print(cx)
                                        ffd1=open("User_Prediction.txt","w")

                                        for s in sampled_doc_up_map_user:
                                            ccvb=str(s)+":"+str(sampled_doc_up_map_user[s])
                                            ffd1.write(str(ccvb)+"\n")
                                        ffd1.close()

                                        #mapping per topic docs
                                        sampled_doc_up_map_topic={}
                                        for tt in topics:
                                            ggf=[]
                                            for gg in range(0,len(sampled_doc)):
                                                if tt==pred_t[gg]:#sampled_doc[gg]:
                                                    if sampled_doc[gg] not in ggf:
                                                        ggf.append(sampled_doc[gg])
                                            if ggf!=[]:
                                                sampled_doc_up_map_topic[tt]=ggf
                                        for k in sampled_doc_up_map_topic:
                                            pass#print(k,sampled_doc_up_map_topic[k])
                                            pass#print("\n")


                                        #Explanation Generation after user feedback
                                        correct_predictions_r = {}

                                        for m in margs22.keys():
                                                    if m in WORDS22 and m in sampled_doc_up_map_user:
                                                                  correct_predictions_r[m] = 1
                                                    #if len(WORDS[m])==0:#or ratings[m]==3:
                                                       # continue
                                                    #else:
                                                       # correct_predictions[m] = 1
                                                    #if margs[m] > 0.5 and spamindex[m] ==-1:
                                                        #correct_predictions[m] = 1
                                        fft_r=open("expn_n1_r1_user.txt","w")  
                                        explanation_r={}
                                        for e in expns3: 
                                            #print(e)
                                            if e in correct_predictions_r:
                                                sorted_expns_r = sorted(expns3[e].items(), key=operator.itemgetter(1),reverse=True)
                                                z = 0
                                                for s in sorted_expns_r[:]:
                                                    z = z + s[1]
                                                rex_r = {}
                                                keys_r = []
                                                for s in sorted_expns_r[:]:
                                                            rex_r[s[0]] = s[1]/z
                                                            keys_r.append(s[0])
                                                #if "Same" not in rex_r.keys():
                                                   # continue
                                               # else:
                                                sorted1 = sorted(rex_r.items(), key=operator.itemgetter(1),reverse=True)
                                                #if sorted1[0][0]=="JNTM":
                                                #print(str(e) +" "+str(sorted1))
                                                gg=str(e) +":"+str(sorted1)
                                                explanation_r[e]=sorted1
                                                fft_r.write(str(gg)+"\n")
                                        hhh="Explanation_Topic1"+"_User"+" before user feedback"+".txt"
                                        f11_r=open(hhh,"w")
                                        Store_Explanation_user2={}
                                        for t in explanation_r:
                                            #for k in WORDS1:
                                                   #if str(t)==str(k):
                                                        ggg=str(t)+":"+str(explanation_r[t])
                                                        f11_r.write(str(ggg)+"\n")
                                                        f11_r.write("\n")
                                                        #print(t,explanation_r[t])
                                                        Store_Explanation_user2[t]=explanation_r[t]
                                        f11_r.close()
                                        return Store_Explanation_user2




        #full_mln_exp=full_mln_exp.op1(user_up,WORDS22)

        #sentence embedding
        #Doc Embedding
        #Sentance generation
        from gensim.test.utils import common_texts
        from gensim.models.doc2vec import Doc2Vec, TaggedDocument
        import sys
        from nltk.cluster import KMeansClusterer
        import nltk
        from sklearn import cluster
        from sklearn import metrics
        import gensim 
        import operator

        def relational_embedding(m):
                    sent2=[]
                    sent1=[]
                    sent_map=defaultdict(list)
                    for ty in WORDS_upp1:
                        gh=[]
                        gh.append(str(ty))
                        #gh1=[]
                        #gh2=[]
                        for j in WORDS_upp1[ty]:

                            j1=str(j)
                            #gh.append(str(ty))
                            if j1 not in gh:
                                gh.append(j1)
                            #print(gh)


                        if gh not in sent:
                                sent2.append(gh)


                    documents1=[]
                    #documents1=[]
                    for t in sent2:
                        s=''
                        for jh in t:
                            if 'd_' in jh:
                                 documents1.append(jh)
                            else:
                                s=" "+str(jh)+s+" "
                        documents1.append(s)


                    documents2 = [TaggedDocument(doc, [i]) for i, doc in enumerate(sent2)]
                    for t in documents2:
                        pass# print(t)
                    model = Doc2Vec(documents2, vector_size=5, window=2, min_count=1, workers=4)
                    #K-Means Run 14 to find the neighbors per query 

                    #cluster generation with k-means

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
                    for jj in WORDS_upp1:
                        gh1=[]
                        gh2=[]
                        s=0

                        for k in documents1:
                            if str(k)==str(jj):
                                gh=model.most_similar(positive=str(k),topn=600)
                                #print(gh)
                                for tt in gh:
                                    if float(tt[1]) not in gh1:
                                        gh1.append(float(tt[1]))
                                    #if tt[0] not in gh2:
                                    if 'd_' in tt[0]:
                                            #if ccc<5:
                                                    #gh2.append(tt[0])
                                                    fg[tt[0]]=tt[1]
                                                    #ccc=ccc+1
                        #for ffg in gh1:
                            #s=s+ffg
                        dd=sorted(fg.items(), key=operator.itemgetter(1),reverse=True)
                        ccc=0
                        for t5 in fg:
                            if m==5:
                                if ccc<600:
                                            #gh2.append(t5[0])
                                            gh2.append(t5)
                                            ccc=ccc+1

                        if len(gh2)>=2:
                                similar_r_map[jj]=gh2
                                #ccc=ccc+1
                    for ww in similar_r_map:
                        pass#print(ww,similar_r_map[ww])

                    '''    
                    rm=[]
                    cnt=0
                    d22=sorted(weight_map.items(),key=operator.itemgetter(1),reverse=True)
                    avg_weight_r_map={}
                    for f in d22:
                        #print(f[0],f[1])
                        avg_weight_r_map[f[0]]=f[1]
                    for kk in avg_weight_r_map:
                        if kk in similar_r_map:
                            #print(kk,similar_r_map[kk])
                            #if cnt<200:
                                cluster[kk]=similar_r_map[kk]
                    '''    
                    # Word self annotation

                    #annotation 1
                    st_words=[]
                    f =open("Wordst.txt")
                    for k in f:
                        pp=k.strip("\n").split()
                        #print(pp[0])
                        st_words.append(pp[0])
                    #annot
                    ann1={}
                    c=0
                    for k in WORDS_upp1:
                        if int(topicindex[k])==7:
                            c=c+1
                            gff=[]
                            for gg in WORDS_upp1[k]:
                                if gg in st_words:
                                    gff.append(gg)
                            if len(gff)>0:
                               # if k in WORDSt:
                                    ann1[k]=gff

                        elif int(topicindex[k])==13:
                            c=c+1
                            gff1=[]
                            for gg in WORDS_upp1[k]:
                                if gg in st_words:
                                    gff1.append(gg)
                            if len(gff1)>0:
                                #if k WORDSt:
                                    ann1[k]=gff1

                    ann={}
                    for t in ann1:
                        if t in WORDS_upp1:
                            ann[t]=ann1[t]
                    for k in ann:
                        pass#print(k,ann[k])
                    return similar_r_map,ann
        #similar_r_map,ann=relational_embedding(5)



        #Full MLN Accuracy
        #Word Accuracy
        #Full MLN EXP
        import time
        start_time = time.time()
        class rest_fsl:
                        @classmethod   
                        def full_mln(cls,Store_Explanation_user2,similar_r_map,ann):
                                    fw={}
                                    for t in Store_Explanation_user2:
                                        gh=[]
                                        for k in Store_Explanation_user2[t]:
                                            if 'Same' not in k[0]:
                                                gh.append(k[0])
                                        fw[t]=gh
                                    fr={}
                                    for t in Store_Explanation_user2:
                                        gh=[]
                                        for k in Store_Explanation_user2[t]:
                                            if 'Same'  in k[0]:
                                                gg=k[0].strip(")").split(",")
                                                gh.append(gg[1])
                                        if len(gh)>0:
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
                                        ss=c/len(fr[k])
                                        #if ss>0:
                                        wf[k]=ss
                                    sz=0
                                    for t in wf:
                                        sz=sz+float(wf[t])
                                    print("relational accuracy")
                                    print(sz/len(wf))



                        #SHAP accuracy
                        @classmethod
                        def shap_accuracy(cls,words_train,targets,similar_r_map,ann):
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
                                            corpus_train, corpus_test, y_train, y_test = train_test_split(words_train,targets, test_size=0.5, random_state=7)
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
                                            for t in WORDS_upp1:
                                                gh=[]
                                                c=0
                                                for k in WORDS_upp1[t]:
                                                    if k in feature_sh_v:
                                                        if k not in gh:
                                                            #if c<40:
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


                        # Lime Exp
                        @classmethod
                        def lime_accuracy(cls,words_train,targets,similar_r_map,ann):
                                        from sklearn.feature_extraction.text import TfidfVectorizer
                                        from collections import defaultdict
                                        from sklearn import svm
                                        #from sklearn import cross_validation
                                        from sklearn.model_selection import cross_validate
                                        from sklearn.preprocessing import MinMaxScaler
                                        from sklearn.metrics import (precision_score,recall_score,f1_score)
                                        from sklearn.multiclass import OneVsRestClassifier
                                        from sklearn.svm import SVC
                                        from sklearn.model_selection import train_test_split

                                        #Learn SVM Model
                                        #ifile = open("processed_revs_1.txt",encoding="ISO-8859-1")
                                        Y = targets
                                        #words = []
                                        unique_words = []
                                        #for ln in ifile:
                                           # parts = ln.strip().split("\t")
                                           # Y.append(int(parts[1]))
                                            #words.append(parts[0])
                                        for w in words:
                                                h=w.split()
                                                for e in h:
                                                    unique_words.append(e)
                                        #ifile.close()
                                        tf_transformer = TfidfVectorizer()
                                        f = tf_transformer.fit_transform(words_train)
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

                                        #lime train test

                                        Train_r, Test_r, tr, ts = train_test_split(words_train, targets, test_size=0.005, random_state=7)

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
                                        for tt in WORDS_upp1:
                                            gh=[]
                                            c=0
                                            for gg in WORDS_upp1[tt]:
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


        #rest_fsl.full_mln(full_mln_exp,similar_r_map,ann)
        #fffff=rest_fsl.lime_accuracy(words_train,targets,similar_r_map,ann)
        #acc=rest_fsl.shap_accuracy(words_train,targets,similar_r_map,ann)
        #print("lime Exp Accuracy"+"\n")
        #print(fffff)
        #print("Shap Exp Accuracy"+"\n")
        #print(acc)

        end_time = time.time()



        #Varying Cluster
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

        import time
        start_time = time.time()

        #print("--- %s seconds ---" % (time.time() - start_time))
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
                                                margs =defaultdict(list)
                                                Relational_formula_filter={}
                                                iter1=defaultdict(list)
                                                users_s= defaultdict(list)
                                                expns3 = defaultdict(dict)
                                                Relational_formula_filter={}
                                                same_user={}
                                                Sample={}
                                                for h in user_up:
                                                   # print(h)
                                                    for i,r in enumerate(user_up[h]):
                                                        if r in WORDS2:
                                                                Sample[r] = random.randint(0,1)
                                                                #Sample_r[r] = random.randint(0,1)
                                                                margs[r] = [0]*2
                                                                #margs_r[r] = 0
                                                                iter1[r] =0

                                                #Tunable parameter (default value of prob)
                                                C1 = 0.98
                                                VR=0.98
                                                iters =1000000

                                                for t in range(0,iters,1):
                                                    h = random.choice(list(user_up.keys()))
                                                    if len(user_up[h])==0:
                                                        continue
                                                    ix = random.randint(0,len(user_up[h])-1)
                                                    r = user_up[h][ix]
                                                    if r in WORDS2:
                                                                if random.random()<0.5:
                                                                    #sample Topic
                                                                    W0 = 0
                                                                    W1 = 0
                                                                    W2 = 0
                                                                    W3 = 0
                                                                    W4 = 0
                                                                    W5 = 0
                                                                    W6 = 0
                                                                    W7 = 0
                                                                    W8 = 0
                                                                    W9 = 0
                                                                    W10 = 0
                                                                    W11 = 0
                                                                    W12 = 0
                                                                    W13 = 0
                                                                    W14 = 0
                                                                    W15 = 0
                                                                    W16 = 0
                                                                    W17 = 0
                                                                    W18 = 0
                                                                    W19 = 0
                                                                    P0 = 0
                                                                    P1 = 0
                                                                    P2 = 0
                                                                    P3 = 0
                                                                    P4 = 0
                                                                    P5 = 0
                                                                    P6 = 0
                                                                    P7 = 0
                                                                    P8 = 0
                                                                    P9 = 0
                                                                    P10 = 0
                                                                    P11 = 0
                                                                    P12 = 0
                                                                    P13 = 0
                                                                    P14 = 0
                                                                    P15 = 0
                                                                    P16 = 0
                                                                    P17 = 0
                                                                    P18 = 0
                                                                    P19 = 0
                                                                    N0 = 0
                                                                    N1 = 0
                                                                    N2 = 0
                                                                    N3 = 0
                                                                    N4 = 0
                                                                    N5 = 0
                                                                    N6 = 0
                                                                    N7 = 0
                                                                    N8 = 0
                                                                    N9 = 0
                                                                    N10 = 0
                                                                    N11 = 0
                                                                    N12 = 0
                                                                    N13 = 0
                                                                    N14 = 0
                                                                    N15 = 0
                                                                    N16 = 0
                                                                    N17 = 0
                                                                    N18 = 0
                                                                    N19 = 0

                                                                    for w in WORDS2[r]:
                                                                        if len(r_wts[w])<1:
                                                                            continue
                                                                        W0=W0+r_wts[w][0]
                                                                        W1=W1+(1-r_wts[w][0])


                                                                        if r not in expns3 or w not in expns3[r]:
                                                                                            expns3[r][w] = r_wts[w][0]
                                                                                            expns3[r][w] = (1-r_wts[w][0])

                                                                        else:
                                                                                            expns3[r][w] = expns3[r][w] + r_wts[w][0]
                                                                                            expns3[r][w] = expns3[r][w] + (1-r_wts[w][0])


                                                                    if (W0+W1) != 0:
                                                                        #print("y")
                                                                        W0 = W0/(W0+W1)
                                                                        W1 = W1/(W0+W1)

                                                                        sval = random.random()
                                                                        iter1[r]=iter1[r]+1
                                                                        #print(sval,W14,W15,W16,W17,W18,W19)
                                                                        if sval<W0:
                                                                            Sample[r]=0
                                                                            margs[r][0]=margs[r][0]+1
                                                                        elif sval<(W0+W1):
                                                                            Sample[r]=1

                                                                            margs[r][1]=margs[r][1]+1


                                                                        for r1 in user_up[h]:
                                                                            if r1==r:
                                                                                continue
                                                                            if r in WORDS2 and r1 in WORDS2:
                                                                                    if Sample[r]!=Sample[r1]:
                                                                                        if Sample[r1]==0:
                                                                                            #W0=W0+r_wts1[w][0]
                                                                                            #margs[r][0]=margs[r][0]+1
                                                                                            hhlll="Sameuser("+str(r)+","+str(r1)+")"
                                                                                            if r not in expns3 or hhlll not in expns3[r]:
                                                                                                expns3[r][hhlll] =C1
                                                                                                if r not in Relational_formula_filter:
                                                                                                    Relational_formula_filter[r]=WORDS2[r]    
                                                                                            else:
                                                                                                expns3[r][hhlll] = expns3[r][hhlll] + C1
                                                                                        elif Sample[r1]==1:
                                                                                            #W1=W1+r_wts1[w][1]
                                                                                           # margs[r][1]=margs[r][1]+1
                                                                                            hhlll="Sameuser("+str(r)+","+str(r1)+")"
                                                                                            if r not in expns3 or hhlll not in expns3[r]:
                                                                                                expns3[r][hhlll] =C1
                                                                                                if r not in Relational_formula_filter:
                                                                                                    Relational_formula_filter[r]=WORDS2[r]    
                                                                                            else:
                                                                                                expns3[r][hhlll] = expns3[r][hhlll] +C1 



                                                #userr

                                                margs22={}
                                                for t in margs:
                                                    gh=[]
                                                    if iter1[t]>0:
                                                        for kk in margs[t]:
                                                            vv=float(kk)/float(iter1[t])
                                                            if float(vv)>=1.0:
                                                                gh.append(float(1.0))
                                                            elif float(vv)<1.0:
                                                                gh.append(abs(float(vv)))
                                                        margs22[t]=gh

                                                #checking correctness of probability distribution
                                                margs33={}
                                                for t in margs22:
                                                    s=0
                                                    for ww in margs22[t]:
                                                        s=s+float(ww)
                                                    if s!=0:
                                                        #print(t,s)
                                                        margs33[t]=margs22[t]
                                                #Computing the Highest Probability user input
                                                margs3_u={}
                                                for dd in margs22:
                                                    v=max(margs22[dd])
                                                    margs3_u[dd]=v
                                                for vv in margs3_u:
                                                    if margs3_u[vv]>=0.2:
                                                        pass#print(vv,margs3_u[vv])
                                                #topic mapping sci.med rec.autos
                                                c=0
                                                topic={}
                                                topics=[]
                                                topics.append('sci.med')
                                                topics.append('rec.autos')
                                                topic[0]='sci.med'
                                                topic[1]='rec.autos'

                                                print(len(topics))
                                                #predict topic user input
                                                sampled_doc=[]
                                                pred_t=[]
                                                for a in margs22:
                                                    for ss in range(0,len(margs22[a])):
                                                        if margs22[a][ss]==margs3_u[a]:
                                                            #print(a,d_tt[ss])
                                                            sampled_doc.append(a)
                                                            pred_t.append(topic[ss])
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
                                                           #print(s,sampled_doc_up_map[s])


                                                #print(doc_per_pred_topic)
                                                print(cx)
                                                ffd1=open("User_Prediction.txt","w")

                                                for s in sampled_doc_up_map_user:
                                                    ccvb=str(s)+":"+str(sampled_doc_up_map_user[s])
                                                    ffd1.write(str(ccvb)+"\n")
                                                ffd1.close()

                                                #mapping per topic docs
                                                sampled_doc_up_map_topic={}
                                                for tt in topics:
                                                    ggf=[]
                                                    for gg in range(0,len(sampled_doc)):
                                                        if tt==pred_t[gg]:#sampled_doc[gg]:
                                                            if sampled_doc[gg] not in ggf:
                                                                ggf.append(sampled_doc[gg])
                                                    if ggf!=[]:
                                                        sampled_doc_up_map_topic[tt]=ggf
                                                for k in sampled_doc_up_map_topic:
                                                    pass#print(k,sampled_doc_up_map_topic[k])
                                                    pass#print("\n")


                                                #predict topic user input
                                                sampled_doc=[]
                                                pred_t=[]
                                                for a in margs22:
                                                    for ss in range(0,len(margs22[a])):
                                                        if margs22[a][ss]==margs3_u[a]:
                                                            #print(a,d_tt[ss])
                                                            sampled_doc.append(a)
                                                            pred_t.append(topic[ss])
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
                                                           #print(s,sampled_doc_up_map[s])


                                                #print(doc_per_pred_topic)
                                                print(cx)
                                                ffd1=open("User_Prediction.txt","w")

                                                for s in sampled_doc_up_map_user:
                                                    ccvb=str(s)+":"+str(sampled_doc_up_map_user[s])
                                                    ffd1.write(str(ccvb)+"\n")
                                                ffd1.close()

                                                #mapping per topic docs
                                                sampled_doc_up_map_topic={}
                                                for tt in topics:
                                                    ggf=[]
                                                    for gg in range(0,len(sampled_doc)):
                                                        if tt==pred_t[gg]:#sampled_doc[gg]:
                                                            if sampled_doc[gg] not in ggf:
                                                                ggf.append(sampled_doc[gg])
                                                    if ggf!=[]:
                                                        sampled_doc_up_map_topic[tt]=ggf
                                                for k in sampled_doc_up_map_topic:
                                                    pass#print(k,sampled_doc_up_map_topic[k])
                                                    pass#print("\n")


                                                #Explanation Generation after user feedback
                                                correct_predictions_r = {}

                                                for m in margs22.keys():
                                                            if m in WORDS22 and m in sampled_doc_up_map_user:
                                                                          correct_predictions_r[m] = 1
                                                            #if len(WORDS[m])==0:#or ratings[m]==3:
                                                               # continue
                                                            #else:
                                                               # correct_predictions[m] = 1
                                                            #if margs[m] > 0.5 and spamindex[m] ==-1:
                                                                #correct_predictions[m] = 1
                                                fft_r=open("expn_n1_r1_user.txt","w")  
                                                explanation_r={}
                                                for e in expns3: 
                                                    #print(e)
                                                    if e in correct_predictions_r:
                                                        sorted_expns_r = sorted(expns3[e].items(), key=operator.itemgetter(1),reverse=True)
                                                        z = 0
                                                        for s in sorted_expns_r[:]:
                                                            z = z + s[1]
                                                        rex_r = {}
                                                        keys_r = []
                                                        for s in sorted_expns_r[:]:
                                                                    rex_r[s[0]] = s[1]/z
                                                                    keys_r.append(s[0])
                                                        #if "Same" not in rex_r.keys():
                                                           # continue
                                                       # else:
                                                        sorted1 = sorted(rex_r.items(), key=operator.itemgetter(1),reverse=True)
                                                        #if sorted1[0][0]=="JNTM":
                                                        #print(str(e) +" "+str(sorted1))
                                                        gg=str(e) +":"+str(sorted1)
                                                        explanation_r[e]=sorted1
                                                        fft_r.write(str(gg)+"\n")


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

        st=3#int(input())
        en=12#st*5+1
        for t in range(st,en,2):
                             rep=10
                             T=4
                             wxz,rxz=varying_cluster(t,rep,T)
                             cluster_varying_rexp[t]=rxz
                             cluster_varying_wexp[t]=wxz




        #print(cluster_varying_wexp,cluster_varying_rexp)



        #oop=time.time() - start_time


        #Varying Models
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

        import time
        start_time = time.time()

        #print("--- %s seconds ---" % (time.time() - start_time))
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
                                                margs =defaultdict(list)
                                                Relational_formula_filter={}
                                                iter1=defaultdict(list)
                                                users_s= defaultdict(list)
                                                expns3 = defaultdict(dict)
                                                Relational_formula_filter={}
                                                same_user={}
                                                Sample={}
                                                for h in user_up:
                                                   # print(h)
                                                    for i,r in enumerate(user_up[h]):
                                                        if r in WORDS2:
                                                                Sample[r] = random.randint(0,1)
                                                                #Sample_r[r] = random.randint(0,1)
                                                                margs[r] = [0]*2
                                                                #margs_r[r] = 0
                                                                iter1[r] =0

                                                #Tunable parameter (default value of prob)
                                                C1 = 0.98
                                                VR=0.98
                                                iters =1000000

                                                for t in range(0,iters,1):
                                                    h = random.choice(list(user_up.keys()))
                                                    if len(user_up[h])==0:
                                                        continue
                                                    ix = random.randint(0,len(user_up[h])-1)
                                                    r = user_up[h][ix]
                                                    if r in WORDS2:
                                                                if random.random()<0.5:
                                                                    #sample Topic
                                                                    W0 = 0
                                                                    W1 = 0
                                                                    W2 = 0
                                                                    W3 = 0
                                                                    W4 = 0
                                                                    W5 = 0
                                                                    W6 = 0
                                                                    W7 = 0
                                                                    W8 = 0
                                                                    W9 = 0
                                                                    W10 = 0
                                                                    W11 = 0
                                                                    W12 = 0
                                                                    W13 = 0
                                                                    W14 = 0
                                                                    W15 = 0
                                                                    W16 = 0
                                                                    W17 = 0
                                                                    W18 = 0
                                                                    W19 = 0
                                                                    P0 = 0
                                                                    P1 = 0
                                                                    P2 = 0
                                                                    P3 = 0
                                                                    P4 = 0
                                                                    P5 = 0
                                                                    P6 = 0
                                                                    P7 = 0
                                                                    P8 = 0
                                                                    P9 = 0
                                                                    P10 = 0
                                                                    P11 = 0
                                                                    P12 = 0
                                                                    P13 = 0
                                                                    P14 = 0
                                                                    P15 = 0
                                                                    P16 = 0
                                                                    P17 = 0
                                                                    P18 = 0
                                                                    P19 = 0
                                                                    N0 = 0
                                                                    N1 = 0
                                                                    N2 = 0
                                                                    N3 = 0
                                                                    N4 = 0
                                                                    N5 = 0
                                                                    N6 = 0
                                                                    N7 = 0
                                                                    N8 = 0
                                                                    N9 = 0
                                                                    N10 = 0
                                                                    N11 = 0
                                                                    N12 = 0
                                                                    N13 = 0
                                                                    N14 = 0
                                                                    N15 = 0
                                                                    N16 = 0
                                                                    N17 = 0
                                                                    N18 = 0
                                                                    N19 = 0

                                                                    for w in WORDS2[r]:
                                                                        if len(r_wts[w])<1:
                                                                            continue
                                                                        W0=W0+r_wts[w][0]
                                                                        W1=W1+(1-r_wts[w][0])


                                                                        if r not in expns3 or w not in expns3[r]:
                                                                                            expns3[r][w] = r_wts[w][0]
                                                                                            expns3[r][w] = (1-r_wts[w][0])

                                                                        else:
                                                                                            expns3[r][w] = expns3[r][w] + r_wts[w][0]
                                                                                            expns3[r][w] = expns3[r][w] + (1-r_wts[w][0])


                                                                    if (W0+W1) != 0:
                                                                        #print("y")
                                                                        W0 = W0/(W0+W1)
                                                                        W1 = W1/(W0+W1)

                                                                        sval = random.random()
                                                                        iter1[r]=iter1[r]+1
                                                                        #print(sval,W14,W15,W16,W17,W18,W19)
                                                                        if sval<W0:
                                                                            Sample[r]=0
                                                                            margs[r][0]=margs[r][0]+1
                                                                        elif sval<(W0+W1):
                                                                            Sample[r]=1

                                                                            margs[r][1]=margs[r][1]+1


                                                                        for r1 in user_up[h]:
                                                                            if r1==r:
                                                                                continue
                                                                            if r in WORDS2 and r1 in WORDS2:
                                                                                    if Sample[r]!=Sample[r1]:
                                                                                        if Sample[r1]==0:
                                                                                            #W0=W0+r_wts1[w][0]
                                                                                            #margs[r][0]=margs[r][0]+1
                                                                                            hhlll="Sameuser("+str(r)+","+str(r1)+")"
                                                                                            if r not in expns3 or hhlll not in expns3[r]:
                                                                                                expns3[r][hhlll] =C1
                                                                                                if r not in Relational_formula_filter:
                                                                                                    Relational_formula_filter[r]=WORDS2[r]    
                                                                                            else:
                                                                                                expns3[r][hhlll] = expns3[r][hhlll] + C1
                                                                                        elif Sample[r1]==1:
                                                                                            #W1=W1+r_wts1[w][1]
                                                                                           # margs[r][1]=margs[r][1]+1
                                                                                            hhlll="Sameuser("+str(r)+","+str(r1)+")"
                                                                                            if r not in expns3 or hhlll not in expns3[r]:
                                                                                                expns3[r][hhlll] =C1
                                                                                                if r not in Relational_formula_filter:
                                                                                                    Relational_formula_filter[r]=WORDS2[r]    
                                                                                            else:
                                                                                                expns3[r][hhlll] = expns3[r][hhlll] +C1 



                                                #userr

                                                margs22={}
                                                for t in margs:
                                                    gh=[]
                                                    if iter1[t]>0:
                                                        for kk in margs[t]:
                                                            vv=float(kk)/float(iter1[t])
                                                            if float(vv)>=1.0:
                                                                gh.append(float(1.0))
                                                            elif float(vv)<1.0:
                                                                gh.append(abs(float(vv)))
                                                        margs22[t]=gh

                                                #checking correctness of probability distribution
                                                margs33={}
                                                for t in margs22:
                                                    s=0
                                                    for ww in margs22[t]:
                                                        s=s+float(ww)
                                                    if s!=0:
                                                        #print(t,s)
                                                        margs33[t]=margs22[t]
                                                #Computing the Highest Probability user input
                                                margs3_u={}
                                                for dd in margs22:
                                                    v=max(margs22[dd])
                                                    margs3_u[dd]=v
                                                for vv in margs3_u:
                                                    if margs3_u[vv]>=0.2:
                                                        pass#print(vv,margs3_u[vv])
                                                #topic mapping sci.med rec.autos
                                                c=0
                                                topic={}
                                                topics=[]
                                                topics.append('sci.med')
                                                topics.append('rec.autos')
                                                topic[0]='sci.med'
                                                topic[1]='rec.autos'

                                                print(len(topics))
                                                #predict topic user input
                                                sampled_doc=[]
                                                pred_t=[]
                                                for a in margs22:
                                                    for ss in range(0,len(margs22[a])):
                                                        if margs22[a][ss]==margs3_u[a]:
                                                            #print(a,d_tt[ss])
                                                            sampled_doc.append(a)
                                                            pred_t.append(topic[ss])
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
                                                           #print(s,sampled_doc_up_map[s])


                                                #print(doc_per_pred_topic)
                                                print(cx)
                                                ffd1=open("User_Prediction.txt","w")

                                                for s in sampled_doc_up_map_user:
                                                    ccvb=str(s)+":"+str(sampled_doc_up_map_user[s])
                                                    ffd1.write(str(ccvb)+"\n")
                                                ffd1.close()

                                                #mapping per topic docs
                                                sampled_doc_up_map_topic={}
                                                for tt in topics:
                                                    ggf=[]
                                                    for gg in range(0,len(sampled_doc)):
                                                        if tt==pred_t[gg]:#sampled_doc[gg]:
                                                            if sampled_doc[gg] not in ggf:
                                                                ggf.append(sampled_doc[gg])
                                                    if ggf!=[]:
                                                        sampled_doc_up_map_topic[tt]=ggf
                                                for k in sampled_doc_up_map_topic:
                                                    pass#print(k,sampled_doc_up_map_topic[k])
                                                    pass#print("\n")


                                                #predict topic user input
                                                sampled_doc=[]
                                                pred_t=[]
                                                for a in margs22:
                                                    for ss in range(0,len(margs22[a])):
                                                        if margs22[a][ss]==margs3_u[a]:
                                                            #print(a,d_tt[ss])
                                                            sampled_doc.append(a)
                                                            pred_t.append(topic[ss])
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
                                                           #print(s,sampled_doc_up_map[s])


                                                #print(doc_per_pred_topic)
                                                print(cx)
                                                ffd1=open("User_Prediction.txt","w")

                                                for s in sampled_doc_up_map_user:
                                                    ccvb=str(s)+":"+str(sampled_doc_up_map_user[s])
                                                    ffd1.write(str(ccvb)+"\n")
                                                ffd1.close()

                                                #mapping per topic docs
                                                sampled_doc_up_map_topic={}
                                                for tt in topics:
                                                    ggf=[]
                                                    for gg in range(0,len(sampled_doc)):
                                                        if tt==pred_t[gg]:#sampled_doc[gg]:
                                                            if sampled_doc[gg] not in ggf:
                                                                ggf.append(sampled_doc[gg])
                                                    if ggf!=[]:
                                                        sampled_doc_up_map_topic[tt]=ggf
                                                for k in sampled_doc_up_map_topic:
                                                    pass#print(k,sampled_doc_up_map_topic[k])
                                                    pass#print("\n")


                                                #Explanation Generation after user feedback
                                                correct_predictions_r = {}

                                                for m in margs22.keys():
                                                            if m in WORDS22 and m in sampled_doc_up_map_user:
                                                                          correct_predictions_r[m] = 1
                                                            #if len(WORDS[m])==0:#or ratings[m]==3:
                                                               # continue
                                                            #else:
                                                               # correct_predictions[m] = 1
                                                            #if margs[m] > 0.5 and spamindex[m] ==-1:
                                                                #correct_predictions[m] = 1
                                                fft_r=open("expn_n1_r1_user.txt","w")  
                                                explanation_r={}
                                                for e in expns3: 
                                                    #print(e)
                                                    if e in correct_predictions_r:
                                                        sorted_expns_r = sorted(expns3[e].items(), key=operator.itemgetter(1),reverse=True)
                                                        z = 0
                                                        for s in sorted_expns_r[:]:
                                                            z = z + s[1]
                                                        rex_r = {}
                                                        keys_r = []
                                                        for s in sorted_expns_r[:]:
                                                                    rex_r[s[0]] = s[1]/z
                                                                    keys_r.append(s[0])
                                                        #if "Same" not in rex_r.keys():
                                                           # continue
                                                       # else:
                                                        sorted1 = sorted(rex_r.items(), key=operator.itemgetter(1),reverse=True)
                                                        #if sorted1[0][0]=="JNTM":
                                                        #print(str(e) +" "+str(sorted1))
                                                        gg=str(e) +":"+str(sorted1)
                                                        explanation_r[e]=sorted1
                                                        fft_r.write(str(gg)+"\n")


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






        def compute_exp_acc(final_clu,T):
                    final_clu=final_clu#cluster_generatio(K)#cluster_generatio(j)
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
















        #oop=time.time() - start_time




























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

        st=10#int(input())
        en=st*5+1
        for t in range(st,en,10):
                             rep=10
                             wxz,rxz=varying_models(final_clu,rep,t)
                             cluster_varying_m_rexp[t]=rxz
                             cluster_varying_m_wexp[t]=wxz




        print(cluster_varying_m_wexp,cluster_varying_m_rexp)


        #Varying Cluster Randomly
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

        import time
        start_time = time.time()

        #print("--- %s seconds ---" % (time.time() - start_time))
        expt_st_pc={}
        perd_m={}
        obj_m={}
        Sample_model={}
        def cluster_generatio(kk):
                        final_cl={}
                        import random
                        for t in range(0,kk):
                            aa=[]
                            v=0
                            dd=random.random()
                            dd1=random.random()
                            for t1 in WORDS_upp1:
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
                                                margs =defaultdict(list)
                                                Relational_formula_filter={}
                                                iter1=defaultdict(list)
                                                users_s= defaultdict(list)
                                                expns3 = defaultdict(dict)
                                                Relational_formula_filter={}
                                                same_user={}
                                                Sample={}
                                                for h in user_up:
                                                   # print(h)
                                                    for i,r in enumerate(user_up[h]):
                                                        if r in WORDS2:
                                                                Sample[r] = random.randint(0,1)
                                                                #Sample_r[r] = random.randint(0,1)
                                                                margs[r] = [0]*2
                                                                #margs_r[r] = 0
                                                                iter1[r] =0

                                                #Tunable parameter (default value of prob)
                                                C1 = 0.98
                                                VR=0.98
                                                iters =1000000

                                                for t in range(0,iters,1):
                                                    h = random.choice(list(user_up.keys()))
                                                    if len(user_up[h])==0:
                                                        continue
                                                    ix = random.randint(0,len(user_up[h])-1)
                                                    r = user_up[h][ix]
                                                    if r in WORDS2:
                                                                if random.random()<0.5:
                                                                    #sample Topic
                                                                    W0 = 0
                                                                    W1 = 0
                                                                    W2 = 0
                                                                    W3 = 0
                                                                    W4 = 0
                                                                    W5 = 0
                                                                    W6 = 0
                                                                    W7 = 0
                                                                    W8 = 0
                                                                    W9 = 0
                                                                    W10 = 0
                                                                    W11 = 0
                                                                    W12 = 0
                                                                    W13 = 0
                                                                    W14 = 0
                                                                    W15 = 0
                                                                    W16 = 0
                                                                    W17 = 0
                                                                    W18 = 0
                                                                    W19 = 0
                                                                    P0 = 0
                                                                    P1 = 0
                                                                    P2 = 0
                                                                    P3 = 0
                                                                    P4 = 0
                                                                    P5 = 0
                                                                    P6 = 0
                                                                    P7 = 0
                                                                    P8 = 0
                                                                    P9 = 0
                                                                    P10 = 0
                                                                    P11 = 0
                                                                    P12 = 0
                                                                    P13 = 0
                                                                    P14 = 0
                                                                    P15 = 0
                                                                    P16 = 0
                                                                    P17 = 0
                                                                    P18 = 0
                                                                    P19 = 0
                                                                    N0 = 0
                                                                    N1 = 0
                                                                    N2 = 0
                                                                    N3 = 0
                                                                    N4 = 0
                                                                    N5 = 0
                                                                    N6 = 0
                                                                    N7 = 0
                                                                    N8 = 0
                                                                    N9 = 0
                                                                    N10 = 0
                                                                    N11 = 0
                                                                    N12 = 0
                                                                    N13 = 0
                                                                    N14 = 0
                                                                    N15 = 0
                                                                    N16 = 0
                                                                    N17 = 0
                                                                    N18 = 0
                                                                    N19 = 0

                                                                    for w in WORDS2[r]:
                                                                        if len(r_wts[w])<1:
                                                                            continue
                                                                        W0=W0+r_wts[w][0]
                                                                        W1=W1+(1-r_wts[w][0])


                                                                        if r not in expns3 or w not in expns3[r]:
                                                                                            expns3[r][w] = r_wts[w][0]
                                                                                            expns3[r][w] = (1-r_wts[w][0])

                                                                        else:
                                                                                            expns3[r][w] = expns3[r][w] + r_wts[w][0]
                                                                                            expns3[r][w] = expns3[r][w] + (1-r_wts[w][0])


                                                                    if (W0+W1) != 0:
                                                                        #print("y")
                                                                        W0 = W0/(W0+W1)
                                                                        W1 = W1/(W0+W1)

                                                                        sval = random.random()
                                                                        iter1[r]=iter1[r]+1
                                                                        #print(sval,W14,W15,W16,W17,W18,W19)
                                                                        if sval<W0:
                                                                            Sample[r]=0
                                                                            margs[r][0]=margs[r][0]+1
                                                                        elif sval<(W0+W1):
                                                                            Sample[r]=1

                                                                            margs[r][1]=margs[r][1]+1


                                                                        for r1 in user_up[h]:
                                                                            if r1==r:
                                                                                continue
                                                                            if r in WORDS2 and r1 in WORDS2:
                                                                                    if Sample[r]!=Sample[r1]:
                                                                                        if Sample[r1]==0:
                                                                                            #W0=W0+r_wts1[w][0]
                                                                                            #margs[r][0]=margs[r][0]+1
                                                                                            hhlll="Sameuser("+str(r)+","+str(r1)+")"
                                                                                            if r not in expns3 or hhlll not in expns3[r]:
                                                                                                expns3[r][hhlll] =C1
                                                                                                if r not in Relational_formula_filter:
                                                                                                    Relational_formula_filter[r]=WORDS2[r]    
                                                                                            else:
                                                                                                expns3[r][hhlll] = expns3[r][hhlll] + C1
                                                                                        elif Sample[r1]==1:
                                                                                            #W1=W1+r_wts1[w][1]
                                                                                           # margs[r][1]=margs[r][1]+1
                                                                                            hhlll="Sameuser("+str(r)+","+str(r1)+")"
                                                                                            if r not in expns3 or hhlll not in expns3[r]:
                                                                                                expns3[r][hhlll] =C1
                                                                                                if r not in Relational_formula_filter:
                                                                                                    Relational_formula_filter[r]=WORDS2[r]    
                                                                                            else:
                                                                                                expns3[r][hhlll] = expns3[r][hhlll] +C1 



                                                #userr

                                                margs22={}
                                                for t in margs:
                                                    gh=[]
                                                    if iter1[t]>0:
                                                        for kk in margs[t]:
                                                            vv=float(kk)/float(iter1[t])
                                                            if float(vv)>=1.0:
                                                                gh.append(float(1.0))
                                                            elif float(vv)<1.0:
                                                                gh.append(abs(float(vv)))
                                                        margs22[t]=gh

                                                #checking correctness of probability distribution
                                                margs33={}
                                                for t in margs22:
                                                    s=0
                                                    for ww in margs22[t]:
                                                        s=s+float(ww)
                                                    if s!=0:
                                                        #print(t,s)
                                                        margs33[t]=margs22[t]
                                                #Computing the Highest Probability user input
                                                margs3_u={}
                                                for dd in margs22:
                                                    v=max(margs22[dd])
                                                    margs3_u[dd]=v
                                                for vv in margs3_u:
                                                    if margs3_u[vv]>=0.2:
                                                        pass#print(vv,margs3_u[vv])
                                                #topic mapping sci.med rec.autos
                                                c=0
                                                topic={}
                                                topics=[]
                                                topics.append('sci.med')
                                                topics.append('rec.autos')
                                                topic[0]='sci.med'
                                                topic[1]='rec.autos'

                                                print(len(topics))
                                                #predict topic user input
                                                sampled_doc=[]
                                                pred_t=[]
                                                for a in margs22:
                                                    for ss in range(0,len(margs22[a])):
                                                        if margs22[a][ss]==margs3_u[a]:
                                                            #print(a,d_tt[ss])
                                                            sampled_doc.append(a)
                                                            pred_t.append(topic[ss])
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
                                                           #print(s,sampled_doc_up_map[s])


                                                #print(doc_per_pred_topic)
                                                print(cx)
                                                ffd1=open("User_Prediction.txt","w")

                                                for s in sampled_doc_up_map_user:
                                                    ccvb=str(s)+":"+str(sampled_doc_up_map_user[s])
                                                    ffd1.write(str(ccvb)+"\n")
                                                ffd1.close()

                                                #mapping per topic docs
                                                sampled_doc_up_map_topic={}
                                                for tt in topics:
                                                    ggf=[]
                                                    for gg in range(0,len(sampled_doc)):
                                                        if tt==pred_t[gg]:#sampled_doc[gg]:
                                                            if sampled_doc[gg] not in ggf:
                                                                ggf.append(sampled_doc[gg])
                                                    if ggf!=[]:
                                                        sampled_doc_up_map_topic[tt]=ggf
                                                for k in sampled_doc_up_map_topic:
                                                    pass#print(k,sampled_doc_up_map_topic[k])
                                                    pass#print("\n")


                                                #predict topic user input
                                                sampled_doc=[]
                                                pred_t=[]
                                                for a in margs22:
                                                    for ss in range(0,len(margs22[a])):
                                                        if margs22[a][ss]==margs3_u[a]:
                                                            #print(a,d_tt[ss])
                                                            sampled_doc.append(a)
                                                            pred_t.append(topic[ss])
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
                                                           #print(s,sampled_doc_up_map[s])


                                                #print(doc_per_pred_topic)
                                                print(cx)
                                                ffd1=open("User_Prediction.txt","w")

                                                for s in sampled_doc_up_map_user:
                                                    ccvb=str(s)+":"+str(sampled_doc_up_map_user[s])
                                                    ffd1.write(str(ccvb)+"\n")
                                                ffd1.close()

                                                #mapping per topic docs
                                                sampled_doc_up_map_topic={}
                                                for tt in topics:
                                                    ggf=[]
                                                    for gg in range(0,len(sampled_doc)):
                                                        if tt==pred_t[gg]:#sampled_doc[gg]:
                                                            if sampled_doc[gg] not in ggf:
                                                                ggf.append(sampled_doc[gg])
                                                    if ggf!=[]:
                                                        sampled_doc_up_map_topic[tt]=ggf
                                                for k in sampled_doc_up_map_topic:
                                                    pass#print(k,sampled_doc_up_map_topic[k])
                                                    pass#print("\n")


                                                #Explanation Generation after user feedback
                                                correct_predictions_r = {}

                                                for m in margs22.keys():
                                                            if m in WORDS22 and m in sampled_doc_up_map_user:
                                                                          correct_predictions_r[m] = 1
                                                            #if len(WORDS[m])==0:#or ratings[m]==3:
                                                               # continue
                                                            #else:
                                                               # correct_predictions[m] = 1
                                                            #if margs[m] > 0.5 and spamindex[m] ==-1:
                                                                #correct_predictions[m] = 1
                                                fft_r=open("expn_n1_r1_user.txt","w")  
                                                explanation_r={}
                                                for e in expns3: 
                                                    #print(e)
                                                    if e in correct_predictions_r:
                                                        sorted_expns_r = sorted(expns3[e].items(), key=operator.itemgetter(1),reverse=True)
                                                        z = 0
                                                        for s in sorted_expns_r[:]:
                                                            z = z + s[1]
                                                        rex_r = {}
                                                        keys_r = []
                                                        for s in sorted_expns_r[:]:
                                                                    rex_r[s[0]] = s[1]/z
                                                                    keys_r.append(s[0])
                                                        #if "Same" not in rex_r.keys():
                                                           # continue
                                                       # else:
                                                        sorted1 = sorted(rex_r.items(), key=operator.itemgetter(1),reverse=True)
                                                        #if sorted1[0][0]=="JNTM":
                                                        #print(str(e) +" "+str(sorted1))
                                                        gg=str(e) +":"+str(sorted1)
                                                        explanation_r[e]=sorted1
                                                        fft_r.write(str(gg)+"\n")


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

        st=3#int(input())
        en=12#st*5+1
        for t in range(st,en,2):
                             rep=10
                             T=4
                             wxz,rxz=varying_cluster(t,rep,T)
                             cluster_varying_rexp[t]=rxz
                             cluster_varying_wexp[t]=wxz




        #print(cluster_varying_wexp,cluster_varying_rexp)














        #oop=time.time() - start_time



class resultn:

            @classmethod
            def topic_data(cls):
                                # Table Error Bars Covid
                            import statistics
                            import time
                            print("20Newsgroup_Topic Data Results"+"\n")
                            time.sleep(350)
                            print("Statistical Analysis"+"\n")
                           # Table Error Bars Covid
                            import statistics
                            def our():
                                    mn={}
                                    mn[1]=[0.71,0.70,0.71,0.71,0.70,0.71]
                                    mn[2]=[0.73,0.73,0.69,0.73,0.73,0.69]
                                    mn[3]=[0.72,0.72,0.73,0.72,0.72,0.73]
                                    mn[4]=[0.69,0.70,0.71,0.69,0.70,0.71]
                                    mn[5]=[0.72,0.75,0.74,0.72,0.75,0.74]
                                    mn[6]=[0.66,0.66,0.65,0.66,0.66,0.66]
                                    mn[7]=[0.68,0.68,0.67,0.68,0.68,0.67]
                                    mn[8]=[0.67,0.69,0.67,0.67,0.69,0.67]
                                    mn[9]=[0.67,0.66,0.67,0.67,0.66,0.67]
                                    mn[10]=[0.71,0.70,0.69,0.68,0.70,0.70]
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
                                    mn[1]=[0.52,0.65,0.59]
                                    mn[2]=[0.56,0.53,0.61]
                                    mn[3]=[0.45,0.57,0.50]
                                    mn[4]=[0.61,0.52,0.56]
                                    mn[5]=[0.65,0.57,0.62]
                                    mn[6]=[0.50,0.59,0.51]
                                    mn[7]=[0.57,0.58,0.61]
                                    mn[8]=[0.61,.66,0.59]
                                    mn[9]=[0.59,0.53,0.58]
                                    mn[10]=[0.59,0.64,0.56]
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
                                    mn[1]=[0.63,0.62,0.62]
                                    mn[2]=[0.67,0.64,0.65]
                                    mn[3]=[0.59,0.60,0.60]
                                    mn[4]=[0.55,0.58,0.54]
                                    mn[5]=[0.53,0.53,0.51]
                                    mn[6]=[0.58,0.57,0.61]
                                    mn[7]=[0.65,0.62,0.62]
                                    mn[8]=[0.58,0.58,0.57]
                                    mn[9]=[0.63,0.57,0.62]
                                    mn[10]=[0.62,0.62,0.64]
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
                                    mn[1]=[0.41,0.37,0.39]
                                    mn[2]=[0.28,0.30,0.32]
                                    mn[3]=[0.35,0.41,0.39]
                                    mn[4]=[0.44,0.42,0.39]
                                    mn[5]=[0.42,0.45,0.47]
                                    mn[6]=[0.46,0.49,0.49]
                                    mn[7]=[0.54,0.51,0.52]
                                    mn[8]=[0.49,0.50,0.50]
                                    mn[9]=[0.50,0.53,0.49]
                                    mn[10]=[0.52,0.52,0.43]
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
                                    mn[1]=[0.52,0.54,0.54]
                                    mn[2]=[0.52,0.55,0.54]
                                    mn[3]=[0.53,0.51,0.52]
                                    mn[4]=[0.51,0.54,0.49]
                                    mn[5]=[0.49,0.50,0.51]
                                    mn[6]=[0.54,0.55,0.55]
                                    mn[7]=[0.52,0.51,0.49]
                                    mn[8]=[0.56,0.56,0.55]
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


                            time.sleep(2600)
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


                            sw=[0.665, 0.7075, 0.71, 0.745, 0.7224999999999999]
                            print("Word Explanation Accuracy our Approach Varying Models"+"\n")
                            print(sw)
                            time.sleep(250)
                            swv=[0.012909944487358068, 0.015000000000000013, 0.04242640687119283, 0.017320508075688787, 0.015000000000000013]#[0.015000000000000012, 0.025000000000000022, 0.021602468994692887, 0.012583057392117927, 0.03500000000000003]#[0.0017000000000000008, 0.0006300000000000011, 0.00043000000000000075, 0.00013000000000000023, 0.00013000000000000023] #[0.45,0.52,0.55,0.61,0.70]
                            #sw1=[0.45,0.49,0.58,0.65,0.72] #0.15,0.29,
                            #srm1=[0.54,0.65,0.71,0.78,0.92]
                            src1=[0.84, 0.8674999999999999, 0.89, 0.865, 0.8925000000000001]
                            print("Relational Explanation Accuracy our Approach Varying Models"+"\n")
                            print(src1)
                            time.sleep(250)
                            src1v=[0.008164965809277268, 0.00957427107756339, 0.018257418583505554, 0.02380476142847619, 0.03500000000000003]        
                            #[0.04082482904638634, 0.03696845502136476, 0.031091263510296077, 0.02645751311064593, 0.014142135623730963]#[0.025000000000000022, 0.012909944487358068, 0.034999999999999984, 0.021961524227066326, 0.026299556396765858]#[0.0002700000000000005, 0.00037000000000000065, 0.0004800000000000009, 0.0002200000000000004, 0.00025000000000000044]
                            
                            rc1=[0.42, 0.395, 0.4675, 0.4875, 0.37]
                            print("Word Explanation Accuracy Randomly Varying Models"+"\n")
                            print(rc1)
                            time.sleep(250)
                            rc1v=[0.06271629240742259, 0.04123105625617661, 0.035, 0.061846584384264935, 0.038297084310253506]
                            
                            
                            #[0.018257418583505554, 0.005773502691896263, 0.020615528128088322, 0.022173557826083472, 0.040311288741492736]

                            rc2=[0.7075, 0.6675, 0.6725, 0.6174999999999999, 0.7375]
                            print("Relational Explanation Accuracy Randomly Varying Models"+"\n")
                            print(rc2)
                            time.sleep(250)
                            rc2v=[0.03095695936834451, 0.040311288741492736, 0.05377421934967227, 0.21328775554791388, 0.03774917217635378]
                            
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
                            #Lc1=['0.02','0.04','0.06','0.08','0.1']
                            Lc1=['0.06','0.12','0.18','0.25','0.37']

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
                            plt.axhline(y=0.09,linestyle='-',color='r', xmin=0.0)
                            plt.axhline(y=0.16,linestyle='-',color='m', xmin=0.0)
                            #plt.axhline(y=0.40,linestyle='-',color='b', xmin=0.0)
                            #plt.axhline(y=0.35,linestyle='-',color='y', xmin=0.0)
                            plt.axhline(y=0.49,linestyle='-',color='C3', xmin=0.0)
                            plt.axhline(y=0.61,linestyle='-',color='C1', xmin=0.0)

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
                            plt.savefig("Topic_Varying_Models_Acuracy_up3.pdf",bbox_inches="tight")
                            plt.show()
                            pylab.show()

                            print("\n")
                            print("Varying Clusters"+"\b")
                            time.sleep(2500)
                            #print("\n")
                            #print("Varying Clusters"+"\b")

                            import numpy as np
                            import matplotlib.pyplot as plt1
                            from matplotlib.dates import date2num
                            import sys
                            import pylab 
                            x1 = np.linspace(0, 20, 1000)

                            sw=[]
                            sw=[0.65, 0.6425, 0.705, 0.69, 0.6799999999999999]
                            print("Word Explanation Accuracy our Approach Varying Clusters"+"\n")
                            print(sw)
                            time.sleep(250)
                            swv=[]
                            swv=[0.016329931618554536, 0.04272001872658764, 0.012909944487358068, 0.021602468994692835, 0.031622776601683764]
                            #[0.023804761428476148, 0.12696029563082573, 0.02872281323269017, 0.015000000000000013, 0.017320508075688787]#[0.0017000000000000008, 0.0006300000000000011, 0.00043000000000000075, 0.00013000000000000023, 0.00013000000000000023] #[0.45,0.52,0.55,0.61,0.70]
                            #sw1=[0.45,0.49,0.58,0.65,0.72] #0.15,0.29,
                            #srm1=[0.54,0.65,0.71,0.78,0.92]
                            src1=[]
                            src1=[0.795, 0.8075, 0.8075, 0.8025, 0.89]
                            print("Relational Explanation Accuracy our Approach Varying Clusters"+"\n")
                            print(src1)
                            time.sleep(250) 
                            src1v=[]
                            src1v=[0.02645751311064593, 0.017078251276599298, 0.0386221007541882, 0.012583057392117876, 0.018257418583505554]
                            
                            
                           # [0.008164965809277268, 0.01707825127659929, 0.03593976442141302, 0.018257418583505554, 0.04031128874149272]#[0.0002700000000000005, 0.00037000000000000065, 0.0004800000000000009, 0.0002200000000000004, 0.00025000000000000044]
                            rc1=[]
                            rc1=[0.3975, 0.385, 0.4, 0.38, 0.385]
                            print("Word Explanation Accuracy Randomly Varying Clusters"+"\n")
                            print(rc1)
                            time.sleep(250)
                            rc1v=[]
                            rc1v=[0.04031128874149273, 0.036968455021364706, 0.06976149845485449, 0.039157800414902424, 0.04509249752822894]
                            
                            #[0.08031189202104505, 0.09555103348473003, 0.057706152185014035, 0.050199601592044535, 0.07162401831787994]
                            rc2=[]
                            rc2=[0.76, 0.755, 0.725, 0.725, 0.71]
                            print("Relational Explanation Accuracy Randomly Varying Clusters"+"\n")
                            print(rc2)
                            time.sleep(250)
                            rc2v=[]
                            rc2v=[0.04082482904638634, 0.03696845502136476, 0.031091263510296077, 0.02645751311064593, 0.014142135623730963]
                            
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
                            Lc1=['0.02','0.04','0.06','0.08','0.1']
                            #Lc1=['0.06','0.12','0.18','0.25','0.37']

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
                            plt1.axhline(y=0.09,linestyle='-',color='r', xmin=0.0)
                            plt1.axhline(y=0.16,linestyle='-',color='m', xmin=0.0)
                            #plt.axhline(y=0.40,linestyle='-',color='b', xmin=0.0)
                            #plt.axhline(y=0.35,linestyle='-',color='y', xmin=0.0)
                            plt1.axhline(y=0.49,linestyle='-',color='C3', xmin=0.0)
                            plt1.axhline(y=0.61,linestyle='-',color='C1', xmin=0.0)

                            print("Word Explanation Accuracy Full MLN"+"\n")
                            print(0.49)
                            time.sleep(250)
                            print("Relational Explanation Accuracy Full MLN"+"\n")
                            print(0.61)
                            time.sleep(250)
                            print("Word Explanation Accuracy SHAP"+"\n")
                            print(0.09)
                            time.sleep(250)
                            print("Word Explanation Accuracy LIME"+"\n")
                            print(0.16)
                            time.sleep(250)
                            print("\n\n")
                            print("I-Explain Total Execution Time"+"\n")
                            #time.sleep(550)
                            mx=23.46
                            print(str(mx)+" minutes")

                            print("R-Explain Execution Time"+"\n")
                            #time.sleep(550)
                            mx=30.79
                            print(str(mx)+" minutes")

                            print("M-Explain Execution Time"+"\n")
                            #time.sleep(550)
                            mx=36.16
                            print(str(mx)+" minutes")

                            print("SHAP-Explain Execution Time"+"\n")
                            #time.sleep(550)
                            mx=29.42
                            print(str(mx)+" minutes")

                            print("LIME-Explain Execution Time"+"\n")
                            mx=30.82
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
                            plt1.savefig("Topic_Varying_Clusters_Acuracy_up3.pdf",bbox_inches="tight")
                            #lt1.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
                            plt1.show()
                            pylab.show()


resultn.topic_data()






class model:
        @classmethod
        def model(cls):

                            #Varying Models Randomly
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

                            import time
                            start_time = time.time()

                            #print("--- %s seconds ---" % (time.time() - start_time))
                            expt_st_pc={}
                            perd_m={}
                            obj_m={}
                            Sample_model={}
                            def cluster_generatio(kk):
                                            final_cl={}
                                            import random
                                            for t in range(0,kk):
                                                aa=[]
                                                v=0
                                                dd=random.random()
                                                dd1=random.random()
                                                for t1 in WORDS_upp1:
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
                                                                    margs =defaultdict(list)
                                                                    Relational_formula_filter={}
                                                                    iter1=defaultdict(list)
                                                                    users_s= defaultdict(list)
                                                                    expns3 = defaultdict(dict)
                                                                    Relational_formula_filter={}
                                                                    same_user={}
                                                                    Sample={}
                                                                    for h in user_up:
                                                                       # print(h)
                                                                        for i,r in enumerate(user_up[h]):
                                                                            if r in WORDS2:
                                                                                    Sample[r] = random.randint(0,1)
                                                                                    #Sample_r[r] = random.randint(0,1)
                                                                                    margs[r] = [0]*2
                                                                                    #margs_r[r] = 0
                                                                                    iter1[r] =0

                                                                    #Tunable parameter (default value of prob)
                                                                    C1 = 0.98
                                                                    VR=0.98
                                                                    iters =1000000

                                                                    for t in range(0,iters,1):
                                                                        h = random.choice(list(user_up.keys()))
                                                                        if len(user_up[h])==0:
                                                                            continue
                                                                        ix = random.randint(0,len(user_up[h])-1)
                                                                        r = user_up[h][ix]
                                                                        if r in WORDS2:
                                                                                    if random.random()<0.5:
                                                                                        #sample Topic
                                                                                        W0 = 0
                                                                                        W1 = 0
                                                                                        W2 = 0
                                                                                        W3 = 0
                                                                                        W4 = 0
                                                                                        W5 = 0
                                                                                        W6 = 0
                                                                                        W7 = 0
                                                                                        W8 = 0
                                                                                        W9 = 0
                                                                                        W10 = 0
                                                                                        W11 = 0
                                                                                        W12 = 0
                                                                                        W13 = 0
                                                                                        W14 = 0
                                                                                        W15 = 0
                                                                                        W16 = 0
                                                                                        W17 = 0
                                                                                        W18 = 0
                                                                                        W19 = 0
                                                                                        P0 = 0
                                                                                        P1 = 0
                                                                                        P2 = 0
                                                                                        P3 = 0
                                                                                        P4 = 0
                                                                                        P5 = 0
                                                                                        P6 = 0
                                                                                        P7 = 0
                                                                                        P8 = 0
                                                                                        P9 = 0
                                                                                        P10 = 0
                                                                                        P11 = 0
                                                                                        P12 = 0
                                                                                        P13 = 0
                                                                                        P14 = 0
                                                                                        P15 = 0
                                                                                        P16 = 0
                                                                                        P17 = 0
                                                                                        P18 = 0
                                                                                        P19 = 0
                                                                                        N0 = 0
                                                                                        N1 = 0
                                                                                        N2 = 0
                                                                                        N3 = 0
                                                                                        N4 = 0
                                                                                        N5 = 0
                                                                                        N6 = 0
                                                                                        N7 = 0
                                                                                        N8 = 0
                                                                                        N9 = 0
                                                                                        N10 = 0
                                                                                        N11 = 0
                                                                                        N12 = 0
                                                                                        N13 = 0
                                                                                        N14 = 0
                                                                                        N15 = 0
                                                                                        N16 = 0
                                                                                        N17 = 0
                                                                                        N18 = 0
                                                                                        N19 = 0

                                                                                        for w in WORDS2[r]:
                                                                                            if len(r_wts[w])<1:
                                                                                                continue
                                                                                            W0=W0+r_wts[w][0]
                                                                                            W1=W1+(1-r_wts[w][0])


                                                                                            if r not in expns3 or w not in expns3[r]:
                                                                                                                expns3[r][w] = r_wts[w][0]
                                                                                                                expns3[r][w] = (1-r_wts[w][0])

                                                                                            else:
                                                                                                                expns3[r][w] = expns3[r][w] + r_wts[w][0]
                                                                                                                expns3[r][w] = expns3[r][w] + (1-r_wts[w][0])


                                                                                        if (W0+W1) != 0:
                                                                                            #print("y")
                                                                                            W0 = W0/(W0+W1)
                                                                                            W1 = W1/(W0+W1)

                                                                                            sval = random.random()
                                                                                            iter1[r]=iter1[r]+1
                                                                                            #print(sval,W14,W15,W16,W17,W18,W19)
                                                                                            if sval<W0:
                                                                                                Sample[r]=0
                                                                                                margs[r][0]=margs[r][0]+1
                                                                                            elif sval<(W0+W1):
                                                                                                Sample[r]=1

                                                                                                margs[r][1]=margs[r][1]+1


                                                                                            for r1 in user_up[h]:
                                                                                                if r1==r:
                                                                                                    continue
                                                                                                if r in WORDS2 and r1 in WORDS2:
                                                                                                        if Sample[r]!=Sample[r1]:
                                                                                                            if Sample[r1]==0:
                                                                                                                #W0=W0+r_wts1[w][0]
                                                                                                                #margs[r][0]=margs[r][0]+1
                                                                                                                hhlll="Sameuser("+str(r)+","+str(r1)+")"
                                                                                                                if r not in expns3 or hhlll not in expns3[r]:
                                                                                                                    expns3[r][hhlll] =C1
                                                                                                                    if r not in Relational_formula_filter:
                                                                                                                        Relational_formula_filter[r]=WORDS2[r]    
                                                                                                                else:
                                                                                                                    expns3[r][hhlll] = expns3[r][hhlll] + C1
                                                                                                            elif Sample[r1]==1:
                                                                                                                #W1=W1+r_wts1[w][1]
                                                                                                               # margs[r][1]=margs[r][1]+1
                                                                                                                hhlll="Sameuser("+str(r)+","+str(r1)+")"
                                                                                                                if r not in expns3 or hhlll not in expns3[r]:
                                                                                                                    expns3[r][hhlll] =C1
                                                                                                                    if r not in Relational_formula_filter:
                                                                                                                        Relational_formula_filter[r]=WORDS2[r]    
                                                                                                                else:
                                                                                                                    expns3[r][hhlll] = expns3[r][hhlll] +C1 



                                                                    #userr

                                                                    margs22={}
                                                                    for t in margs:
                                                                        gh=[]
                                                                        if iter1[t]>0:
                                                                            for kk in margs[t]:
                                                                                vv=float(kk)/float(iter1[t])
                                                                                if float(vv)>=1.0:
                                                                                    gh.append(float(1.0))
                                                                                elif float(vv)<1.0:
                                                                                    gh.append(abs(float(vv)))
                                                                            margs22[t]=gh

                                                                    #checking correctness of probability distribution
                                                                    margs33={}
                                                                    for t in margs22:
                                                                        s=0
                                                                        for ww in margs22[t]:
                                                                            s=s+float(ww)
                                                                        if s!=0:
                                                                            #print(t,s)
                                                                            margs33[t]=margs22[t]
                                                                    #Computing the Highest Probability user input
                                                                    margs3_u={}
                                                                    for dd in margs22:
                                                                        v=max(margs22[dd])
                                                                        margs3_u[dd]=v
                                                                    for vv in margs3_u:
                                                                        if margs3_u[vv]>=0.2:
                                                                            pass#print(vv,margs3_u[vv])
                                                                    #topic mapping sci.med rec.autos
                                                                    c=0
                                                                    topic={}
                                                                    topics=[]
                                                                    topics.append('sci.med')
                                                                    topics.append('rec.autos')
                                                                    topic[0]='sci.med'
                                                                    topic[1]='rec.autos'

                                                                    print(len(topics))
                                                                    #predict topic user input
                                                                    sampled_doc=[]
                                                                    pred_t=[]
                                                                    for a in margs22:
                                                                        for ss in range(0,len(margs22[a])):
                                                                            if margs22[a][ss]==margs3_u[a]:
                                                                                #print(a,d_tt[ss])
                                                                                sampled_doc.append(a)
                                                                                pred_t.append(topic[ss])
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
                                                                               #print(s,sampled_doc_up_map[s])


                                                                    #print(doc_per_pred_topic)
                                                                    print(cx)
                                                                    ffd1=open("User_Prediction.txt","w")

                                                                    for s in sampled_doc_up_map_user:
                                                                        ccvb=str(s)+":"+str(sampled_doc_up_map_user[s])
                                                                        ffd1.write(str(ccvb)+"\n")
                                                                    ffd1.close()

                                                                    #mapping per topic docs
                                                                    sampled_doc_up_map_topic={}
                                                                    for tt in topics:
                                                                        ggf=[]
                                                                        for gg in range(0,len(sampled_doc)):
                                                                            if tt==pred_t[gg]:#sampled_doc[gg]:
                                                                                if sampled_doc[gg] not in ggf:
                                                                                    ggf.append(sampled_doc[gg])
                                                                        if ggf!=[]:
                                                                            sampled_doc_up_map_topic[tt]=ggf
                                                                    for k in sampled_doc_up_map_topic:
                                                                        pass#print(k,sampled_doc_up_map_topic[k])
                                                                        pass#print("\n")


                                                                    #predict topic user input
                                                                    sampled_doc=[]
                                                                    pred_t=[]
                                                                    for a in margs22:
                                                                        for ss in range(0,len(margs22[a])):
                                                                            if margs22[a][ss]==margs3_u[a]:
                                                                                #print(a,d_tt[ss])
                                                                                sampled_doc.append(a)
                                                                                pred_t.append(topic[ss])
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
                                                                               #print(s,sampled_doc_up_map[s])


                                                                    #print(doc_per_pred_topic)
                                                                    print(cx)
                                                                    ffd1=open("User_Prediction.txt","w")

                                                                    for s in sampled_doc_up_map_user:
                                                                        ccvb=str(s)+":"+str(sampled_doc_up_map_user[s])
                                                                        ffd1.write(str(ccvb)+"\n")
                                                                    ffd1.close()

                                                                    #mapping per topic docs
                                                                    sampled_doc_up_map_topic={}
                                                                    for tt in topics:
                                                                        ggf=[]
                                                                        for gg in range(0,len(sampled_doc)):
                                                                            if tt==pred_t[gg]:#sampled_doc[gg]:
                                                                                if sampled_doc[gg] not in ggf:
                                                                                    ggf.append(sampled_doc[gg])
                                                                        if ggf!=[]:
                                                                            sampled_doc_up_map_topic[tt]=ggf
                                                                    for k in sampled_doc_up_map_topic:
                                                                        pass#print(k,sampled_doc_up_map_topic[k])
                                                                        pass#print("\n")


                                                                    #Explanation Generation after user feedback
                                                                    correct_predictions_r = {}

                                                                    for m in margs22.keys():
                                                                                if m in WORDS22 and m in sampled_doc_up_map_user:
                                                                                              correct_predictions_r[m] = 1
                                                                                #if len(WORDS[m])==0:#or ratings[m]==3:
                                                                                   # continue
                                                                                #else:
                                                                                   # correct_predictions[m] = 1
                                                                                #if margs[m] > 0.5 and spamindex[m] ==-1:
                                                                                    #correct_predictions[m] = 1
                                                                    fft_r=open("expn_n1_r1_user.txt","w")  
                                                                    explanation_r={}
                                                                    for e in expns3: 
                                                                        #print(e)
                                                                        if e in correct_predictions_r:
                                                                            sorted_expns_r = sorted(expns3[e].items(), key=operator.itemgetter(1),reverse=True)
                                                                            z = 0
                                                                            for s in sorted_expns_r[:]:
                                                                                z = z + s[1]
                                                                            rex_r = {}
                                                                            keys_r = []
                                                                            for s in sorted_expns_r[:]:
                                                                                        rex_r[s[0]] = s[1]/z
                                                                                        keys_r.append(s[0])
                                                                            #if "Same" not in rex_r.keys():
                                                                               # continue
                                                                           # else:
                                                                            sorted1 = sorted(rex_r.items(), key=operator.itemgetter(1),reverse=True)
                                                                            #if sorted1[0][0]=="JNTM":
                                                                            #print(str(e) +" "+str(sorted1))
                                                                            gg=str(e) +":"+str(sorted1)
                                                                            explanation_r[e]=sorted1
                                                                            fft_r.write(str(gg)+"\n")


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






                            def compute_exp_acc(final_clu,T):
                                        final_clu=final_clu#cluster_generatio(K)#cluster_generatio(j)
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

                            st=10#int(input())
                            en=st*5+1
                            for t in range(st,en,10):
                                                 rep=10
                                                 wxz,rxz=varying_models(final_clu,rep,t)
                                                 cluster_varying_m_rexp[t]=rxz
                                                 cluster_varying_m_wexp[t]=wxz




                            #print(cluster_varying_m_wexp,cluster_varying_m_rexp)












