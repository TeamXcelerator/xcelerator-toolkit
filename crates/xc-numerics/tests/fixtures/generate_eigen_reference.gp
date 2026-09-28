\\ Independently generated PARI/GP reference; original audit repair code.
\\ Run: gp -q generate_eigen_reference.gp > eigen_reference.json
\\ Exact rational matrices; polrootsreal of the exact characteristic polynomial.
default(realprecision, 2000);
cases = [["dense_integer_3", [5,1,2;1,3,-1;2,-1,4]], ["dirichlet_4", [2,-1,0,0;-1,2,-1,0;0,-1,2,-1;0,0,-1,2]], ["dense_rational_4", [5/3,1/7,-2/9,1/11;1/7,7/2,3/13,-1/17;-2/9,3/13,11/4,2/19;1/11,-1/17,2/19,13/5]], ["signed_dense_5", [-4,1,2,0,-1;1,-2,0,3,1;2,0,1,-2,0;0,3,-2,5,2;-1,1,0,2,7]]];
{
print1("{\"schema_version\":1,\"generator\":\"PARI/GP polrootsreal(charpoly(M))\",\"precision_decimal_digits\":2000,\"pari_version\":\"", version(), "\",\"cases\":[");
for(k=1,#cases,
  if(k>1, print1(","));
  M=cases[k][2]; n=matsize(M)[1]; roots=polrootsreal(charpoly(M));
  if(#roots!=n, error("reference matrix must have n real roots"));
  print1("{\"name\":\"",cases[k][1],"\",\"n\":",n,",\"matrix\":[");
  for(i=1,n, for(j=1,n, if(i>1 || j>1, print1(",")); print1("\"",1.0*M[i,j],"\"")));
  print1("],\"eigenvalues_ascending\":[");
  for(i=1,n, if(i>1, print1(",")); print1("\"",roots[i],"\""));
  print1("]}");
);
print("]}");
}
quit;
