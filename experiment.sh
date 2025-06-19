   #!/bin/bash

   numbers=(1 2 3 4 5 6 7 8 9 10 11 12)

   for i in "${!numbers[@]}"; do
     python experiment_linear_prob.py "${numbers[i]}"
   done
