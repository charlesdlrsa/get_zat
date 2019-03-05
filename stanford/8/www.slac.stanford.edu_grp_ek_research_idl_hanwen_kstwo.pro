name kstwo purpose given an array data1 and an array data2 this routine returns the kolmogorov smirnov statistic d and the significance level prob for the null hypothesis that two given data sets are drawn from the same distribution small values of prob show that the cumulative distribution function of data1 is significantly different from that of data2 the arrays data1 and data2 are modified by being sorted into ascending order adapted from a routine of the same name in numerical recipes in c second edition category math calling sequence kstwo data1 data2 d prob inputs data1 2 first second data array outputs data1 2 original data1 2 array sorted into ascending order d ks statistic prob ks significance level modification history written by han wen august 1996 function probks alam kolmogorov smirnov probability function eps1 0.001 eps2 1.0 e 8 fac 2.0 sum termbf 0.0 a2 2.0 alam 2 for j 1100 do begin term fac exp double a2 j 2 sum sum term if abs term le eps1 termbf or abs term le eps2 sum then return sum fac fac alternating signs in sum termbf abs term endfor return 1.0 end pro kstwo data1 data2 d prob n1 n_elements data1 n2 n_elements data2 j1 j2 1l fn1 fn2 0.0 data1 data1 sort data1 data2 data2 sort data2 en1 float n1 en2 float n2 d 0.0 while j1 le n1 and j2 le n2 do begin if we are not done d1 float data1 j1 1 d2 float data2 j2 1 if d1 le d2 then begin next step is in data1 fn1 j1 en1 j1 j1 1 endif if d2 le d1 then begin next step is in data2 fn2 j2 en2 j2 j2 1 endif dt abs fn2 fn1 if dt gt d then d dt endwhile en sqrt en1 en2 en1 en2 prob probks en 0.12 0.11 en d compute significance end
