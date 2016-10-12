%*******ECEN 5733 Final Project****************************
%*******Neural Network Weed Detection using MBP[1]*********
%******The program read the reflectance data from the *****
%******bands of the sensor and trains the network**********
%******Early Stopping is implemented to make sure that ****
%******the validation errors doesnot go beyond a **********
%******specified level*************************************
%***05/02/2008*************by  Aswin Ramachandran**********
%**********************************************************
clear
clc
clf

%---Learning Rate---
    fprintf( 'Learning Rate');
    alpha = 0.1  %0.1
    lambda = 0.00
    earlyStopCount = 0;
    prev_eSq_v =10;

%****Load the Input File******
  load ./nnm_train.txt
  redData = nnm_train(:,2);
  nir1Data = [nnm_train(:,3) ./ redData]';  
  nir2Data = [nnm_train(:,4) ./ redData]';
  nir3Data = [nnm_train(:,5) ./ redData]';
  pg = [ nir1Data; nir2Data; nir3Data];
  targetData = nnm_train(:,7) ;
 %*******Validate Data*******
 load ./nnm_validate.txt
  redData_v = nnm_validate(:,2);
  nir1Data_v = [nnm_validate(:,3) ./ redData_v]';  
  nir2Data_v = [nnm_validate(:,4) ./ redData_v]';
  nir3Data_v = [nnm_validate(:,5) ./ redData_v]';
  targetData_v = nnm_validate(:,7) ;
  pValidate = [nir1Data_v; nir2Data_v; nir3Data_v];
 %*******Test Data*******
 load ./nnm_test.txt
  redData_t = nnm_test(:,2);
  nir1Data_t = [nnm_test(:,3) ./ redData_t]';  
  nir2Data_t = [nnm_test(:,4) ./ redData_t]';
  nir3Data_t = [nnm_test(:,5) ./ redData_t]';
  targetData_t = nnm_test(:,7) ;
  pTest = [nir1Data_t; nir2Data_t; nir3Data_t];
  
%---Plot the Original Function----
    pa = 1 : length(targetData_t);
    actLine = 0:0.1:0.8; 
    subplot(2,1,2), plot (actLine, actLine); legend('Actual');%scatter(targetData_t, targetData_t,'^b');
    hold on
 
    %-----Randomized First Layer Weights & Bias-------
    fprintf( 'Initial Weights and Biases');
  %****3-8-1******
     W1 = [ -0.5+rand -0.5+rand -0.5+rand -0.5+rand -0.5+rand -0.5+rand -0.5+rand -0.5+rand;  -0.5+rand -0.5+rand -0.5+rand -0.5+rand -0.5+rand -0.5+rand -0.5+rand -0.5+rand;...
              -0.5+rand -0.5+rand -0.5+rand -0.5+rand -0.5+rand -0.5+rand -0.5+rand -0.5+rand]'; %Uniform distribution [-0.5 0.5]
     b1 = [ -0.5+rand -0.5+rand -0.5+rand -0.5+rand -0.5+rand -0.5+rand -0.5+rand -0.5+rand]';  
     W2 = [ -0.5+rand -0.5+rand -0.5+rand -0.5+rand -0.5+rand -0.5+rand -0.5+rand -0.5+rand];
  %-----Randomized Second Layer Bias------
     b2 = [ -0.5+rand ];

    if (lambda == 0) % Save the Weights and Bias on SBP
        W1_initial = W1;
        b1_initial = b1;
        W2_initial = W2;
        b2_initial = b2;
    else      % Reuse the Weights and Bias on MBP
        W1 = W1_initial;
        b1 = b1_initial;
        W2 = W2_initial;
        b2 = b2_initial;
    end
    %-----RandPermutation of Input Training Set-------
    j = randperm(length(targetData)); 
    j_v = randperm(length(targetData_v)); 
    j_t = randperm(length(targetData_t)); 
    %--Set Max. Iterations---
    maxIter = 2000;
    tic
    for train = 1 : maxIter +1   
      eSq = 0; eSq_v = 0; eSq_t = 0;
      % **** Mean Square Error ****
      %if ( train <= maxIter )
          for p = 1 : length(targetData)
             n1 = W1*pg(:,p)+ b1 ;
             a1 = logsig(n1); 
             a2 = poslin( W2 * a1  + b2 ); 
             e = targetData(p) - a2 ;
             eSq = eSq + (e^2);
          end
           eSq = eSq/length(targetData);
         
           %*******Validate Error**********
           for p = 1 : length(targetData_v)
             n1 = W1*pValidate(:,p)+ b1 ;
             a1 = logsig(n1); 
             a2 = poslin( W2 * a1  + b2 ); 
             e = targetData_v(p) - a2 ;
             eSq_v = eSq_v + (e^2);
           end
            eSq_v = eSq_v/length(targetData_v);
          %********Use Validate Error for Early Stopping********
           if ( train > 200 )
               earlyStopCount = earlyStopCount + 1;
              % fprintf('EarlyStop = %d', earlyStopCount);
               if (earlyStopCount == 50)
                       if ( (prev_eSq_v - eSq_v) < 0 ) 
                            W2 = W2_25;
                            b2 = b2_25;
                            W1 = W1_25;
                            b1 = b1_25;
                          break;
                       end
                  prev_eSq_v = eSq_v;   % Store previous validation error
                  earlyStopCount = 0;  % Reset Early Stopping
                  %----Save the weights and biases-------
                    disp('Saving'); eSq
                  W2_25 = W2;
                  b2_25 = b2;
                  W1_25 = W1;
                  b1_25 = b1;
               end
              
           else
                % ----Initialize the Weights----
              if ( train == 200 )
                  W2_25 = W2;
                  b2_25 = b2;
                  W1_25 = W1;
                  b1_25 = b1;
               end
               prev_eSq_v = eSq_v;   % Store previous validation error
           end
           %*******Test Error**********
           for p = 1 : length(targetData_t)
             n1 = W1*pTest(:,p)+ b1 ;
             a1 = logsig(n1); 
             a2 = poslin( W2 * a1  + b2 ); 
             e = targetData_t(p) - a2 ;
             eSq_t = eSq_t + (e^2);
           end
            eSq_t = eSq_t/length(targetData_t);
            
             if (train == 1 || mod (train, 100) == 0   )
               fprintf( 'Weights and Biases at Iter = %d\n',train);
                fprintf('W1');
                 (W1)
                fprintf('b1')
                 b1
                fprintf('W2')
                 W2
                fprintf('b2')
                 b2
               fprintf( 'Mean Error Square at Iter = %d',train);
               eSq
               eSq_v
               eSq_t
           end
           subplot(2,1,1), 
           xlabel('No. of Iterations');
           ylabel('Mean Square Error');
           title('Convergence Characteristics ');
           loglog(train, eSq, '*r'); hold on
           loglog(train, eSq_v, '*g'); hold on
           loglog(train, eSq_t, '*c'); hold on
           legend('Training Error', 'Validation Error', 'Testing Error');
           %************Train Data**********************
           %  Update only when the error is decreasing
         
      %    if ( earlyStopCount == 0 ) 
              for p = 1 : length(targetData)
                %----Output of the 1st Layer-----------
                n1 = W1*pg(:,j(p))+ b1 ;
                a1 = logsig(n1)  ; 
                %-----Output of the 2nd Layer----------
                n2 = W2 * a1  + b2;
                a2 =  (poslin( n2 )); 
                %a2 =  (logsig( n2 ));
                 %-----Error-----
                    t = targetData(j(p));
                    e = t - a2;
                 %******CALCULATE THE SENSITIVITIES************
                    %-----Derivative of logsig function----
                    %f1 = dlogsig(n1,a1)  
                   % f1 =  [(1-a1(1))*a1(1) 0; 0 (1-a1(2))*a1(2)] ; 
                    f1 = diag((1-a1).*a1); 
                    %-----Derivative of purelin function---
                    f2 = 1;
                    %f2 = diag((1-a2).*a2);
                    %------Last Layer (2nd) Sensitivity----
                    S2 = -2 * f2 * e;
                    S2mbp = ((t)-n2);

                    %------First Layer Sensitivity---------
                    S1 =   f1 *(W2' * S2);
                    S1mbp = f1 * (W2' * S2mbp);
                %******UPDATE THE WEIGHTS**********************
                    %-----Second Layer Weights & Bias------

                    W2 = W2 - (alpha * S2*(a1)') - (alpha * lambda * S2mbp *(a1)');
                    b2 = b2 - alpha * S2 - (alpha * lambda * S2mbp);
                    %-----First Layer Weights & Bias-------
                    W1 = W1 - alpha * S1*(pg(:,j(p)))' - (alpha * lambda * S1mbp *(pg(:,j(p)))');
                    b1 = b1 - alpha * S1 - (alpha * lambda * S1mbp );
               % end
              end
          %end 
          % End of 21 Input Training Sets
           % ********** Function Apporx. *****************
          if (train == 1 || mod (train, 100) == 0 || train == maxIter )
               disp('Trained');
               subplot(2,1,2), 
               xlabel('Actual Fraction of Weeds in 3 sq feet of grass area');
               ylabel('Estimated Fraction of Weeds in 3 sq feet of grass area');
               title('Correlation of Estimated Value with respect to the Actual Function using Standard Backpropagation');
               legend('Estimated');
            for p = 1 : length(targetData_t)
               n1 = W1*pTest(:,p)+ b1 ;       % Test Data
               a1 = logsig(n1)  ; 
               a2(p) =   (poslin( W2 * a1  + b2 )); 
            end %end for
            %scatter(targetData_t, a2); hold on;
          end 
    end
    toc
    %-------End of Iterations------------
    %***Plot of Final Function******
     subplot(2,1,2),
     for p = 1 : length(targetData_t)
               n1 = W1*pTest(:,p)+ b1 ;  % Test Data
               a1 = logsig(n1)  ; 
               a2(p) =   (poslin( W2 * a1  + b2 )); 
     end %end for
      scatter(targetData_t,a2);
      legend('Actual ','Estimate','Location','NorthWest');
     % legend( 'Actual','Location','NorthWest');
    %  legend( 'Estimated Value','Location','NorthWest');
      hold on;
   % End of Training Iterations
    fprintf( 'No. of Iterations = %d\n',train);
   % fprintf('W1');
   % (W1)
  %  fprintf('b1')
    % b1
   % fprintf('W2')
    % W2
  %  fprintf('b2')
   %  b2
    fprintf( 'Final Mean Squared Error at Iter = %d',train);
     eSq

%******************* End of Weed Detection Program ***************

    
