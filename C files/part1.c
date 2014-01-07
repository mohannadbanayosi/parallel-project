//
//  part1.c
//  ParallelComputing
//
//  Created by Mohannad Banayosi on 1/7/14.
//  Copyright (c) 2014 Mohannad Banayosi. All rights reserved.
//

#include <stdio.h>

int main(int argc, const char * argv[])
{
    
    // insert code here...
    printf("lol!\n");
    double fady = 5.0;
    //    printf("%f\n", fady);
    
    
    int n[ 10 ]; /* n is an array of 10 integers */
    int i,j;
    printf("Address: %d\n", &n);
    int M[10] = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    int N[10] = { 9, 8, 7, 6, 5, 4, 3, 2, 1 };
    int P[10] = { 0 };
    
    
    for (j = 0; j < 9; j++ )
    {
        printf("Element[%d] = %d\n", j, M[j] );
    }
    
    
    int Width = 3;
    for (int i = 0; i < Width; ++i) {
        for (int j = 0; j < Width; ++j) {
            double sum = 0;
            for (int k = 0; k < Width; ++k) {
                double a = M[i * Width + k];
                double b = N[k * Width + j];
                sum += a * b;
            }
            //            printf("lol : %f\n", sum);
            P[i*Width+j]=sum;
        }
    }
    
    for (j = 0; j < 9; j++ )
    {
        printf("Element[%d] = %d\n", j, P[j] );
    }
    return 0;
}

