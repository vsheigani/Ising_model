#include <stdio.h>
#include <math.h>
#include <time.h>
#include <cmath>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

using namespace std;

int random_int(int L);

double random_d();

double Energy(std::vector<std::vector<int>> S, int L);

double Summ(std::vector<std::vector<int>> S, int L);

int main()
{
    double E = 0.0, ESum = 0.0, E2Sum = 0.0, MSum = 0.0;
    double SSum = 0.0, S2Sum = 0.0, Sval = 0.0, Ki = 0.0, Cv = 0.0;
    double Savg = 0.0, S2avg =0.0, Eavg = 0.0, E2avg = 0.0;
    int dE = 0, dS = 0;
    double Senergy = 0.0;
    int MCSs[1] = {1000};
    // int MCSs[4] = {100, 1000, 10000};
    double T = 4.0;
    int Ls[1] = {100};
    // int Ls[4] = {4, 12, 36, 100};
    int k1, k1p1, k1m1;
    int k2, k2p1, k2m1;
    int nSamples = 0;
    double Mavg = 0;
    std::vector<double> ex (9, 0.0);

    for(int MCS: MCSs)
    {
        for (int L : Ls)
        {
            cout << "Monte_Carlo sweeps: " << MCS << ", " << "Latice Size: " << L << endl;
            std::ofstream datfile;
            srand(time(NULL));

            int N = L*L;
            std::vector<std::vector<int>> S (L, std::vector<int>(L, 1));

            string filename = "Ising_L" + std::to_string(L) +"_MCS" + std::to_string(MCS) + ".dat";
            datfile.open(filename, std::ios_base::out);

            E = Energy(S, L);
            Sval = Summ(S, L);
            T = 4;
            while (T>=0.5)
            {
                for (int i = 0; i < 9; i+=2) ex[i]=exp(-2.0*((double)i-4.0)/T);
                SSum = 0.0;
                S2Sum = 0.0;
                MSum = 0.0;
                nSamples = 0;
                ESum = 0.0;
                E2Sum = 0.0;
                for(int j = 0; j < MCS; j++)
                {
                    for(int l=0; l < N; l++)
                    {
                        dE = 0.0;
                        dS = 0.0;
                        k1 = random_int(L);
                        k1p1= k1+1;
                        k1m1= k1-1;
                        if(k1m1==-1) k1m1=L-1;
                        if(k1p1==L)  k1p1=0;

                        k2 = random_int(L);
                        k2p1= k2+1;
                        k2m1= k2-1;
                        if(k2m1==-1) k2m1=L-1;
                        if(k2p1==L)  k2p1=0;
                        Senergy = S[k1][k2] * (S[k1p1][k2] + S[k1m1][k2] + S[k1][k2p1] + S[k1][k2m1]);
                        if (ex[(int)Senergy + 4] > random_d())
                        {
                            dS = -2.0 * S[k1][k2];
                            dE = -2.0 * Senergy;
                            S[k1][k2] = -S[k1][k2];

                        }else{
                            dE = 0.0;
                            dS = 0.0;
                        }
                        Sval += dS;
                        E += dE;
                    }
                    if(j > 0.1* (double)MCS)
                    {
                        ESum += E;
                        MSum += fabs(Sval);
                        E2Sum += pow(E, 2);
                        SSum += Sval;
                        S2Sum += pow(Sval, 2);
                        nSamples++;
                    }
                }
                Savg = SSum/(double)nSamples;
                S2avg = S2Sum/(double)nSamples;
                MSum = MSum/(double)MCS;
                // printf("%E\n", MSum);
                Eavg = ESum/(double)nSamples;
                E2avg = E2Sum/(double)nSamples;
                Cv = (E2avg - pow(Eavg, 2))/((double)N*pow(T,2));
                Mavg = MSum/(double)N;
                Ki = (S2avg - pow(Savg,2))/((double)N * T);
                datfile << std::scientific << T << " " << Mavg << " " << Cv << " " << Ki  << " " << -Eavg/N << endl;
                T -= 0.05;
            }
            datfile.close();
        }
    }
    return 0;
}

double random_d(){
    return (double)rand()/(double)RAND_MAX;
}

int random_int(int L){
    return (rand() % L);
}


double Summ(std::vector<std::vector<int>> S, int L){
    double Sval = 0;
    for (int k1 = 0; k1 < L; k1++)
    {
        for (int k2 = 0; k2 < L; k2++)
        {
            Sval += S[k1][k2];
        }
    }
    return Sval;
}

double Energy(std::vector<std::vector<int>> S, int L){
    double Edipole = 0;
    double E = 0;
    int k1p1, k1m1;
    int k2p1, k2m1;
    
    for (int k1 = 0; k1 < L; k1++)
    {
        k1p1= k1+1;
        k1m1= k1-1;
        if(k1==0) k1m1=L-1;
        if(k1==L-1) k1p1=0;
        for (int k2 = 0; k2 < L; k2++)
        {
            k2p1= k2+1;
            k2m1= k2-1;
            if(k2==0) k2m1=L-1;
            if(k2==L-1) k2p1=0;
            Edipole = S[k1][k2] * (S[k1p1][k2] + S[k1m1][k2] + S[k1][k2p1] + S[k1][k2m1]);
            E += Edipole;
        }
    }
    return E/2;
}

