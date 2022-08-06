#!/usr/bin/env python
# coding: utf-8

# # Kredi Notu Sitemi

# Bir bankanın tüketici kredisi departmanı, ev sermayesi kredi limitlerinin onaylanması için karar verme sürecini otomatikleştirmek istiyor. Bunu yapmak için;
# ampirik olarak üretilmiş ve istatistiksel olarak sağlam bir kredi puanlama modeli oluşturulacak. Aynı zamanda, Eşit Kredi Fırsatı Yasası (Equal Credit Opportunity
# Acty'nin tavsiyelerinin göz ardı edilmemesi gerekiyor. Model, mevcut kredi yüklenimi süreci ile kredi verilen son başvuru sahiplerinden toplanan verilere göre
# oluşturulacaktır.

# ## Veri Seti Hakkında

# Ev Sermayesi veri kümesi (HMEQ), 5.960 yeni ev sermayesi kredisi için mevcut durumları ve kredi performans bilgilerini içerir. Hedef (BAD), bir başvuranın
# nihayetinde temerrüde düştüğünü veya ciddi şekilde suçlu olup olmadığını gösteren ikili bir değişkendir. Bu olumsuz sonuç 1.189 durumda (% 20 ) meydana
# geldi. Her başvuru sahibi için 12 girdi değişkeni kaydedilmiştir.			   

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix


# In[4]:


df = pd.read_csv("hmeq.csv")
df_i = pd.read_csv("hmeq.csv")


# In[5]:


df.head() #Veri kümesine bakış


# # Veriyi Anlamak

# In[6]:


df.shape


# In[7]:


df.info()


# In[8]:


df.describe() #Verinin dağıtımı
              #Anamoli Tespiti
              #Veride hiçbir anamoli yok


# In[9]:


df.columns  #Verinin kolonları


# # Kolonların Açıklamaları :
# 
# •BAD : Hedef değer, kredi battı mı batmadı mı.
# •LOAN : Kredi taleb İpotek edilen varlığın toplam değer.
# •MORTDUE : İpotek edilen varlığın toplam değeri.
# •Reason : Kredi talebinin sebebi. (Ev için yada var olan borcu yapılandırmak için)
# •JOB : Meslek.
# •YOJ : Mevcut işte geçirilien süre.
# •DEROG : Geçmişte yaşanan ciddi gecikme sayısı.
# •DELİNQ : Ödenmeyen kredi sayısı.
# •CLAGE : En eski kredinin yaşı. (Ay olarak)
# •NINQ : Kredi sorgu sayısı.
# •CLAGE : Kredi limiti sayısı.
# •CLNO : Kredi limiti sayısı.
# •DEBTINC: Borç/gelir oranı.

# # Çeşitli Ögelerin Dağılımları :

# In[10]:


print(df["BAD"].value_counts())      #"BAD" hedef değişkeninin dağılımı
df["BAD"].value_counts().plot("barh")  #Hedef sınıf biraz dengesiz. 0'ler %80 1'ler %20


# In[ ]:


print(df["REASON"].value_counts()) #Bu nominal bir değer, kullanabileceğimiz şekilde düzelemeliyiz


# In[ ]:


print(df["JOB"].value_counts()) #Yukarıda olduğu gibi, kullanabileceğimiz şekilde düzenlemeliyiz


# In[ ]:


df["LOAN"].plot.hist(bins = 20,figsize=(15,7.5))  #Kredi değişkeninin dağılımı 
                                                  # 10000-30000 arasında yoğunluk yüksek


# In[11]:


df["DEBTINC"].plot.hist(bins = 20,figsize=(15,5)) # Yoğunluk 25-50 civarında yüksek


# In[12]:


df["CLAGE"].plot.hist(bins = 20,figsize=(15,7.5)) # Yoğunluk 100-300 civarında yüksek
                                                #()>= 600) daki değerleri kırparak daha iyi sonuçlar elde edebiliriz


# In[13]:


df["CLNO"].plot.hist(bins = 20,figsize=(15,5)) #Dağılım iyi gözüktüğü için burada hiçbir şeyi değiştirmemize gerek yok


# In[14]:


df["VALUE"].plot.hist(bins = 80,figsize=(15,7.5))  # Yoğunluk 80000-100000 civarında yüksek
                                                # Sonlara doğru düşük veriler var (>= 400000) 
                                                #ortalamaya göre biraz yüksek olan çok daha az değer var, bunları kapatabiliriz


# In[15]:


df["MORTDUE"].plot.hist(bins = 40,figsize=(15,7.5))   # Yoğunluk 40000-100000 civarında yüksek


# In[16]:


df["YOJ"].plot.hist(bins = 40,figsize=(15,7.5))
# Bu çok çarpık, çarpıklığı azaltmak için bu değişkeni değiştirirsek daha iyi olur


# In[17]:


df["DEROG"].value_counts() # Aşağlayıcı alanlar sadece birkaç değerde tespit edildi


# In[18]:


df["DELINQ"].value_counts() # Çoğu sıfır.
# Yukarıdaki durumda olduğu gibi bir binary değişken oluşturmak faydalı olacaktır


# In[19]:


df["NINQ"].value_counts() # Çoğunlukla ilk beş değer arasında dağıtılır


# ## Sonuçlar:
# • Dağılımlar iyi ve verilerde herhangi bir anormallik yok.
# • DEBTİNC'in çok fazla sayıda eksik verisi var (Bir sonraki bölümde, değişkenlerin hesaplanmasında ele alınacaktır).
# • YOJ özelliği oldukça eğridir ve eğriliği azaltmak için değiştirilebilir.
# • Nominal özellikler : JOB ve REASON, lojistik regresyon modeli için kullanabileceğimiz şekilde değiştirilmelidir.
# • DELINQ, DEROG yeni ikili değişkenler oluşturmak için 2 sınıfa ayrılabilir.
# DEĞER, İHTİYAÇ, BORÇ, sonunda kapatılabilir, yani çok yüksek değerler seçilen daha düşük bir değere ayarlanır.

# ## Giriş değişkenlerini yerleştirme

# In[25]:


df.isnull().sum()


# In[26]:


# Nominal değişkenler
# Çoğunluk sınıfı kullanılarak değiştirme
# JOB değişkeni durumunda çoğunluk sınıfı Diğer
# REASON değişkeni durumunda çoğunluk sınıfı DebtCon'dur

df["REASON"].fillna(value = "DebtCon",inplace = True)
df["JOB"].fillna(value = "Other",inplace = True)


# In[27]:


df["DEROG"].fillna(value=0,inplace=True)
df["DELINQ"].fillna(value=0,inplace=True)


# In[33]:


# Atlanmış bir şey olup olmadığını kontrol ediliyor
# Gördüğünüz gibi tüm eksik değerler dolduruldu

df.isnull().sum()


# ### Verilere Eksik değerleri doldurduktan sonra son bakış 

# In[34]:


df.head()


# # İmputasyondan sonra modellerin verilere uygulanması

# Değiştirme/imputasyon sonrası verilere temel Sınıflandırmanın uygulanması. Hem Lojistik Regresyon hem de Karar ağacı algoritmalarını uygulayarak performansı kontrol edelim.
# Algoritmaları uygulamadan önce, veriler 2:1 oranında yani test verileri %33 ve eğitim verileri %67 oranında eğitim ve test kümelerine bölünür.
# Ayrıca JOB,REASON dışındaki tüm sütunları girdi özniteliği olarak alarak (nominal öznitelikler olduklarından, bir sonraki bölümde ele alınacak olan kullanılabilir olmaları için diğer değişkenlere dönüştürülmeleri gerekir).

# In[39]:


# modülleri içe aktarma
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# BAD,JOB,REASON özelliklerini giriş setinden kaldırma
x_basic = df.drop(columns=["BAD","JOB","REASON"])
y = df["BAD"]

# Veriyi into test ve eğitim setlerine bölme
x_basic_tr,x_basic_te,y_tr,y_te = train_test_split(x_basic,y,test_size =.33,random_state=1)
logreg_basic = LogisticRegression()

# Eğitim seti ile temel lojistik regresyon modelinin eğitimi
logreg_basic.fit(x_basic_tr,y_tr)

# Coefficients yazdırma
print("intercept ")
print(logreg_basic.intercept_)
print("")
print("coefficients ")
print(logreg_basic.coef_)

# Yukarıda oluşturulan algoritmayı kullanarak test senaryolarının çıktısını tahmin etme
y_pre = logreg_basic.predict(x_basic_te)

print("accuracy score : ",a1)
print("f1 score : ",f1)
print("precision score : ",p1)
print("recall score : ",r1)


# In[41]:


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    Bu fonksiyon, karışıklık matrisini yazdırır ve çizer.
    Normalizasyon, `normalize=True` ayarlanarak uygulanabilir.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[43]:


# Yukarıdaki algoritma için Karışıklık matrisinin hesaplanması

cnf_matrix = confusion_matrix(y_te, y_pre)
np.set_printoptions(precision=2)

# Normalleştirilmemiş karışıklık matrisini çizme
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=["BAD"],
                      title='Confusion matrix - Logistic Regression Algorithm')

plt.show()


# In[45]:


# Gerekli modülleri içe aktarma
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

dectree_basic = DecisionTreeClassifier()
dectree_basic.max_depth = 100
# Temel Karar Ağacı modelini eğitim seti ile eğitmek
dectree_basic.fit(x_basic_tr,y_tr)

# Predicting the output of the test cases using the algorithm created above
y_pre = dectree_basic.predict(x_basic_te)

# 
Yukarıda oluşturulan algoritmayı kullanarak test senaryolarının çıktısını tahmin etmek

a2 = accuracy_score(y_te,y_pre)
f2 = f1_score(y_te, y_pre, average="macro")
p2 = precision_score(y_te, y_pre, average="macro")
r2 = recall_score(y_te, y_pre, average="macro")
print("accuracy score : ",a2)
print("f1 score : ",f2)
print("precision score : ",p2)
print("recall score : ",r2)

# Yukarıdaki algoritma için Karışıklık matrisinin hesaplanması

cnf_matrix = confusion_matrix(y_te, y_pre)
np.set_printoptions(precision=2)


# Normalleştirilmemiş karışıklık matrisini çizme
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=["BAD"],
                      title='Confusion matrix,Decision Tree Algorithm')

plt.show()


# In[ ]:




