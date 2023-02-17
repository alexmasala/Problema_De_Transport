import numpy
import matplotlib.pyplot as grafic

def Ex4_ProblemaDeTransport_Grupa1052_Masala_Alexandra(fo,fc,fcost,dim,nmax,pr,pm):
    # rezolvarea problemei generale de transport

    # fenotip: matricea de transport
    # genotip: ordinea in care se aloca elementele din matricea de transport <=> permutare, ind. mereu fezabili
    # I: fo - fisier oferta (text, 1xm)
    #    fc - fisier cerere (text, 1 x n)
    #    fcost - fisier costuri (text, m x n)
    #    dim - dimensiune populatie
    #    nmax - numar maxim de iteratii
    #    pr - probabilitate de recombinare
    #    pm - probabilitate de mutatie
    # E: sol - solutia de transport gasita
    #    cost - cost de transport calculat
    # Exemple de apel:
    #    import Ex4_ProblemaDeTransport_Grupa1052_Masala_Alexandra as GTR
    #    s,c=GTR.Ex4_ProblemaDeTransport_Grupa1052_Masala_Alexandra('T_oferta.txt','T_cerere.txt','T_costuri.txt',10,50,0.8,0.1)
    # Cel mai mic cost gasit: 17000.0
    # Solutia de transport:
    # 40    0   80
    # 60   60    0

    # initializari - parametri de intrare
    oferta=numpy.genfromtxt(fo)
    cerere=numpy.genfromtxt(fc)
    costuri=numpy.genfromtxt(fcost)


    # generare populatie initiala
    pop = gen_pop(dim, oferta, cerere, costuri)
    v = [min(1000. / pop[:, -1])]

    # bucla GA
    ok=True
    t=0
    while t<nmax and ok:
        # selectie parinti
        parinti = s_ruleta_SUS(pop)
        # recombinare
        desc = recombinare(parinti, pr, oferta,cerere,costuri)
        # mutatie
        descm = mutatie(desc, pm, oferta,cerere,costuri)
        # selectie generatie urmatoare
        pop = s_elitista(pop, descm)
        # alte operatii
        # retine cea mai buna solutie
        vmax = min(1000./pop[:,-1])
        i = numpy.argmin(pop[:, -1])
        best = pop[i][:-1]
        v.append(vmax)
        t+=1
        ok=max(pop[:,-1])!=min(pop[:,-1])

    print("Cel mai bun cost găsit: ",vmax)
    print("Soluția de transport:")
    sol=gen_alocare(best,oferta,cerere)
    print(sol)
    fig=grafic.figure()
    grafic.plot(v)
    verificare(sol,oferta,cerere)
    return (sol,vmax)

def f_obiectiv(x,oferta,cerere,costuri):
    # functia obiectiv pentru problema de transport

    # I: x - cromozom evaluat
    #    oferta, cerere - restrictii
    #    costuri - matricea costurilor de transport
    #    E: c - calitate (1000/cost transport)

    a=gen_alocare(x,oferta,cerere)
    c=1000./numpy.sum(a*costuri)
    return c

def gen_alocare(permutare,oferta,cerere):
    # conversie genotip->fenotip = decodificare (alocare resurse cu respectarea restrictiilor)

    # I: permutare - genotip, ordinea de alocare
    #    oferta, cerere - restrictii
    # E: x - alocarea obtinuta, sub forma de matrice

    m=len(oferta)
    n=len(cerere)
    x=numpy.zeros((m,n))
    i=0
    CR=sum(cerere)
    o_r=oferta.copy()   #oferta ramasa
    c_r=cerere.copy()   #cerere ramasa
    while CR>0:
        lin,col=numpy.unravel_index(int(permutare[i]),(m,n))
        x[lin,col]=min([o_r[lin],c_r[col]])
        o_r[lin]-=x[lin,col]
        c_r[col]-=x[lin,col]
        CR-=x[lin,col]
        i+=1
    return x

def gen_pop(dim,oferta,cerere,costuri):
    # generare populatie initiala pentru problema de transport

    # I: dim - dimensiune populatie
    #    oferta - oferta de transport (1xm)
    #    cerere = cerere de transport (1xn)
    #    costuri - costuri de transport (mxn)
    # E: pop - populatia generata

    m=len(oferta)
    n=len(cerere)
    pop=numpy.zeros((dim,m*n+1))
    for i in range(dim):
        x=numpy.random.permutation(m*n)
        pop[i, :-1] = x
        pop[i, -1] = f_obiectiv(x, oferta, cerere, costuri)
    return(pop)

#SELECTIA PARINTILOR
def d_FPS_ss(pop,c):
    # distributia de selectie FPS cu sigma scalare

    # I: pop - bazinul de selectie
    #    c - constanta din formula de ajustare. uzual: 2
    # E: p - vector probabilitati de selectie individuale
    #    q - vector probabilitati de selectie cumulate

    m,n=numpy.shape(pop)
    medie=numpy.mean(pop[:,n-1])
    sigma=numpy.std(pop[:,n-1])
    val=medie-c*sigma
    g=[numpy.max([0, pop[i][n-1]-val]) for i in range(m)]
    s=numpy.sum(g)
    p=g/s
    q=[numpy.sum(p[:i+1]) for i in range(m)]
    return p,q

def s_ruleta_SUS(pop):
    # selectia tip ruleta multibrat

    # I: pop - bazinul de selectie
    # E: rez - populatia selectata

    m,n=numpy.shape(pop)
    p,q=d_FPS_ss(pop,2)                 #sau alta distributie
    rez=pop.copy()
    i=0
    k=0
    r=numpy.random.uniform(0,1/m)
    while k<m:
        while r<=q[i]:
            rez[k,:n]=pop[i,:n]
            r+=1/m
            k+=1
        i+=1
    return rez

def s_elitista(pop,desc):
    # selectia elitista a generatiei urmatoare

    # I: pop - populatia curenta
    #    desc - descendentii populatiei curente
    # E: noua - matricea descendentilor selectati

    noua=desc.copy()
    dim,n=numpy.shape(pop)
    max1=max(pop[:,n-1])
    i=numpy.argmax(pop[:,n-1])
    max2=max(desc[:,n-1])
    if max1>max2:
        k=numpy.argmin(desc[:,n-1])
        noua[k,:]=pop[i,:]
    return noua

#RECOMBINARE
def r_CX(x,y,pr):
    # operatorul de recombinare Cycle Crossover pentru permutari

    # I: x,y - cromozomii parinti
    #    pr - probabilitatea de recombinare
    # E: a,b - descendenti

    a=x.copy()
    b=y.copy()
    r=numpy.random.uniform(0,1)
    if r<pr:
        m=len(x)
        c,nrc=cicluri(x,y)
        for t in range(2,nrc+1,2):
            for i in range(m):
                if c[i]==t:
                    a[i]=y[i]
                    b[i]=x[i]
    return a, b

def cicluri(x,y):
    # determinare cicluri pentru CX

    # I: x, y - cromozomi
    # E: c - vector cu indicii ciclurilor
    #    cite - numarul de cicluri

    m=len(x)
    c=numpy.zeros(m,dtype=int)
    continua=1
    i=0
    cite=1
    while continua:
        a=y[i]
        c[i]=cite
        while x[i]!=a:
            j=list(x).index(a)
            c[j]=cite
            a=y[j]
        try:
            i=list(c).index(0)
            cite+=1
        except:
            continua=0
    return c,cite

def recombinare(parinti,pr,oferta,cerere,costuri):
    # etapa de recombinare

    # I: parinti - multiseutl parintilor
    #    pr - probabilitatea de recombianre
    #    oferta, cerere, costuri - paramatri ai problemei
    # E: desc - descendentii creati

    dim,n=numpy.shape(parinti)
    desc=numpy.zeros((dim,n))
    #selectia aleatoare a perechilor de parinti
    perechi=numpy.random.permutation(dim)
    for i in range(0,dim,2):
        x = parinti[perechi[i], :n - 1]
        y = parinti[perechi[i + 1], :n - 1]
        d1, d2 = r_CX(x, y, pr)
        desc[i, :n - 1] = d1
        desc[i][n - 1] = f_obiectiv(d1, oferta,cerere,costuri)
        desc[i + 1, :n - 1] = d2
        desc[i + 1][n - 1] = f_obiectiv(d2, oferta,cerere,costuri)
    return desc

#MUTATIE
def m_perm_schimb(x,pm):
    # operatorul de mutatie prin interschimbare pentru permutari

    # I: x - individul supus mutatiei
    #    pm - probabilitatea de mutatie
    # E: y - individul rezultat

    y=x.copy()
    r=numpy.random.uniform(0,1)
    if r<pm:
        m = len(x)
        p = numpy.random.randint(0, m, 2)
        while p[0] == p[1]:
            p[1] = numpy.random.randint(0,m)
        p.sort()
        y[p[1]]=x[p[0]]
        y[p[0]]=x[p[1]]
    return y

def mutatie(desc,pm,oferta,cerere,costuri):
    # etapa de mutatie

    # I: desc - descendentii obtinuti in etapa de recombianre
    #    pm - probabilitatea de mutatie
    #    oferta,cerere,costuri - restrictiile problemei
    # E: descm - descendenti modificati

    dim,n=numpy.shape(desc)
    descm=desc.copy()
    for i in range(dim):
        x=descm[i,:n-1]
        y=m_perm_schimb(x,pm)
        descm[i,:n-1]=y
        descm[i,n-1]=f_obiectiv(y,oferta,cerere,costuri)
    return descm

def verificare(sol,oferta,cerere):
    # verificare solutie

    # I: sol - solutia problemei de transport
    #    oferta,cerere - restrictiile problemei
    # E: -

    o_r=oferta-numpy.sum(sol,axis=1)
    c_r=cerere-numpy.sum(sol,axis=0)

    mino=min(o_r); maxo=max(o_r)
    minc=min(c_r); maxc=max(c_r)
    print("Oferta rămasă:", o_r)
    if mino<0:
        print("Eroare la ofertă: se consumă mai mult de cît e disponibil")
    if maxo>0:
        print("O parte din ofertă va rămâne neconsumată.")
    if mino==0 and maxo==0:
        print("Oferta e consumată perfect")

    print("Cerere rămasă:", c_r)
    if minc<0:
        print("Eroare la cerere: se transportă mai mult de cît se cere")
    if maxc>0:
        print("Eroare la cerere: se transportă mai putin de cît se cere")
    if minc==0 and maxc==0:
        print("Cererea acoperită perfect")

