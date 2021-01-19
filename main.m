close all; clear all; clc;
%-------------------------------------------------------------------------%
% 
%   Description: Reimplementation of <U. Rosolia and F. Borrelli, "Learning 
%                Model Predictive Control for Iterative Tasks. A Data-
%                Driven Control Framework," in IEEE Transactions on 
%                Automatic Control, vol. 63, no. 7, pp. 1883-1896, July 
%                2018, doi: 10.1109/TAC.2017.2753460> using CasADi.
% 
%                Make sure CasADi is on your MATLAB search path!
%
%-------------------------------------------------------------------------%

% Define simulation parameters
Sim = par_simulation( 1 );

% Define parameters of dynamic system
Dyn = par_system( 1 );

% Define parameters of LMPC
Lmpc = par_lmpc( 1 );



%% Initial and optimal trajectory
% The LMPC approach requires the initial sampled safe set to be non-empty.
% Therefore, an initial suboptimal solution is acquired by solving the 
% problem with a large prediction horizon and more restrictive bounds.
[x0, u0] = compute_optimal_trajectory( Sim.x0, Sim.N0, Lmpc.Q, Lmpc.R, ...
                                       Dyn.Par.A, Dyn.Par.B, Sim.xMax0, ...
                                       Sim.xMin0, Dyn.Bounds.uMax, ...
                                       Dyn.Bounds.uMin );

% Compute cost-to-go for the initial solution
Q0 = compute_cost( x0, u0, Lmpc.Q, Lmpc.R );

% To obtain an optimal reference solution, the problem is solved using a 
% large prediction horizon without manipulating the bounds.
[xOpt, uOpt] = compute_optimal_trajectory( Sim.x0, Sim.N0, Lmpc.Q, ...
                                           Lmpc.R, Dyn.Par.A, ...
                                           Dyn.Par.B, Dyn.Bounds.xMax, ...
                                           Dyn.Bounds.xMin, ...
                                           Dyn.Bounds.uMax, ...
                                           Dyn.Bounds.uMin );

% Compute cost-to-go for optimal solution
QOpt = compute_cost( xOpt, uOpt, Lmpc.Q, Lmpc.R );

% Prepare parametric solver object that is updated with the current state
% xk and the new state set and Q function
Solver = set_solver( Lmpc.Q, Lmpc.R, Lmpc.N, Dyn.Par.A, Dyn.Par.B, ...
                     Dyn.Bounds.xMin, Dyn.Bounds.xMax, Dyn.Bounds.uMin, ...
                     Dyn.Bounds.uMax, Lmpc.nSafeSet, ...
                     Dyn.Dims.nStates, Dyn.Dims.nInputs, Lmpc.solver, ...
                     Lmpc.expand, Lmpc.printTime );



%% Simulate LMPC
% Initialize safe set and corresponding Q-function
Data.x = [ x0 zeros( Dyn.Dims.nStates, Lmpc.nSafeSet-size(x0,2) ) ];
Data.Q = [ compute_cost( x0, u0, Lmpc.Q, Lmpc.R ) ...
           zeros( 1, Lmpc.nSafeSet-size(x0,2) ) ];

% Run simulation
[X, U, Q] = sim_lmpc( Solver, Sim.x0, Data.x, Data.Q, Dyn.Par.A, ...
                      Dyn.Par.B, Lmpc.Q, Lmpc.R, Sim.nIter, Sim.tol );



%% Visualize results
plot_results( x0, u0, Q0, X, U, Q, xOpt, uOpt, QOpt, Sim.nIter )



%% Auxilary functions
function Sim = par_simulation( printInfo )
% 
%   Signature   : Sim = par_simulation( printInfo )
%
%   Description : Set parameter struct for simulation.
%
%   Parameters  : printInfo -> Bool whether to print information
% 
%   Return      : Sim -> Parameter struct
%
%-------------------------------------------------------------------------%

% Number of iterations
Sim.nIter = 10;

% Numerical tolerance for which origin is considered reached 
% ||x||_2 < tol
Sim.tol = 1e-09;

% Initial state
Sim.x0 = [-14;2];

% Parameters for optimization to obtain initial feasible trajectory
Sim.N0 = 200;
Sim.xMax0 = [15; 2];
Sim.xMin0 = [-15; -2];

if nargin>0
  if printInfo
    disp( 'Information on the simulation parameters' )
    fprintf( 'Number of iterations jMax = %d\n', Sim.nIter )
    fprintf( 'Initial state x0: [%2.2f; %2.2f]\n\n', Sim.x0 )
  end
end

end


function Dyn = par_system( printInfo )
% 
%   Signature   : Dyn = par_system( printInfo )
%
%   Description : Set parameter struct for example system.
%
%   Parameters  : printInfo -> Bool whether to print information
% 
%   Return      : Dyn -> Parameter struct
%
%-------------------------------------------------------------------------%

% Define the system matrices
Dyn.Par.A = [1  1; 0  1];
Dyn.Par.B = [0; 1];

% Define the input constraint set U
uMax = 1.5;

% Define the state constraint set X
xMax = 15.0;
xDotMax = 15.0;
    
% Add information about dynamic system
Dyn.Dims.nStates = 2;
Dyn.Dims.nInputs = 1;
Dyn.Bounds.uMax =  uMax;
Dyn.Bounds.uMin = -uMax;
Dyn.Bounds.xMax = [xMax; xDotMax];
Dyn.Bounds.xMin = [-xMax; -xDotMax];
Dyn.Labels.states = {'x', 'xDot'};
Dyn.Labels.inputs = {'u'};

if nargin>0
  if printInfo
    disp( 'Information on the dynamic system' )
    fprintf( ['System dynamics: x(k+1) = [%d, %d; %d, %d] * x(k) + ' ...
              '[%d;%d] * u(k)\n' ], Dyn.Par.A, Dyn.Par.B )
    fprintf( 'Bounds x: [%2.2f, %2.2f; %2.2f, %2.2f]\n', ...
             Dyn.Bounds.xMax, Dyn.Bounds.xMin )
    fprintf( 'Bounds u: [%2.2f; %2.2f]\n\n', Dyn.Bounds.uMax, ...
              Dyn.Bounds.uMin )
  end
end

end


function Lmpc = par_lmpc( printInfo )
% 
%   Signature   : Lmpc = par_lmpc( printInfo )
%
%   Description : Set parameter struct for LMPC.
%
%   Parameters  : printInfo -> Bool whether to print information
% 
%   Return      : Lmpc -> Parameter struct
%
%-------------------------------------------------------------------------%

% Prediction horizon
Lmpc.N = 4;

% State cost
Lmpc.Q = [1, 0; 0, 1];

% Input cost
Lmpc.R = 1;

% Maximum array size for sampled safe set (keep dimensions of predefiend 
% optimization problem constant)
Lmpc.nSafeSet = 250;

% QP solver ('qpoases', 'osqp', 'qrqp', 'ipopt')
Lmpc.solver = 'qpoases';

% Solver options
Lmpc.expand = 1;
Lmpc.printTime = 0;

if nargin>0
  if printInfo
    disp( 'Information on the LMPC parameters' )
    fprintf( 'Prediction horizon N: %d\n', Lmpc.N )
    fprintf( 'QP solver: %s\n', Lmpc.solver )
    fprintf( 'Weight matrices: Q = [%d, %d; %d, %d], R = %d\n\n', ...
              Lmpc.Q, Lmpc.R )
  end
end

end


function Solver = set_solver( Q, R, N, A, B, xMin, xMax, uMin, uMax, nSafeSet, nStates, nInputs, solver, expand, printTime )
%
%   Signature   : Solver = set_solver( Q, R, N, A, B, xMin, xMax, uMin, uMax, nSafeSet, nStates, nInputs, solver, expand, printTime )
%
%   Description : Set solver for parametric QP within LMPC
% 
%   Parameters  : Q, R -> Weighting matrices for the cost function
%                 N -> Prediction horizon
%                 A, B -> Linear state space matrices
%                 xMax, xMin, uMax, uMin -> Bounds on states and inputs
%                 nSafeSet -> Maximum number of points stored within safe set
%                 nStates, nInputs -> State and input dimensions
%                 solver -> String containing name of QP solver to use
%                 expand -> Bool controlling whether to expand casadi.MX to
%                           casadi.SX
%                 printTime -> Bool controlling whether to print solver
%                              timings
% 
%   Return      : Solver -> 
% 
%-------------------------------------------------------------------------%

% Initialize Opti instance
if any( strcmp( solver, {'qpoases', 'osqp', 'qrqp', 'hpipm'} ) )
  Opti = casadi.Opti('conic');
else
  Opti = casadi.Opti();
end

% Define symbolic variables for states, inputs and multipliers used for the
% convex hull
x_ = Opti.variable( nStates, N+1 );
u_ = Opti.variable( nInputs, N );
lambda_ = Opti.variable( nSafeSet );

% Define symbolic variables for the parameters of the optimization problem
% that will be updated every time step
xk_ = Opti.parameter( nStates );

% Define symbolic variables for the parameters of the optimization problem
% which will be updated after each iteration
xSafeSet_ = Opti.parameter( nStates, nSafeSet );
Q_ = Opti.parameter( nSafeSet );

% Define initial condition
Opti.subject_to( x_(:,1) == xk_ );

% Define dynamic system gap closing constraints
Opti.subject_to( x_(:,2:N+1) == A*x_(:,1:N) + B*u_ );

% Define bounds
Opti.subject_to( xMin <= x_ <= xMax );
Opti.subject_to( uMin <= u_ <= uMax );

% Define constraint to force last state to be inside the convex hull of the
% safe set. The convex hull is defined by (see Borrelli p. )
% 
%   conv(SS) = { sum_i=1^k lambda_i*x_i, x_i in SS, 
%                                        lambda_i>=0, i=1,...,k, 
%                                        sum_i=1^k lambda_i=1}
% The definition requires constraints for ...
% ... sum_i=1^k lambda_i*x_i 
Opti.subject_to( x_(:,N+1) == xSafeSet_*lambda_ );

% ... lambda_i>=0, i=1,...,k, 
Opti.subject_to( lambda_ >= 0 );

% ... sum_i=1^k lambda^i=1
Opti.subject_to( sum(lambda_) == 1 );

% Define sum of stage cost
cost_ = [];
for ii=1:N
    cost_ = [cost_; x_(:,ii).' * Q * x_(:,ii) ...
                    + u_(:,ii).' * R *u_(:,ii) ];
end 

% Define the terminal cost as the cost-to-go corresponding to the reached
% last state inside the convex hull which is defined by the multipliers
% lambda
cost_ = [cost_; Q_.'*lambda_];

% Set cost function
Opti.minimize( sum(cost_) );

% Change MX to SX
OptsSolver.expand = expand;

% Toggle whether casadi prints output timing information after optimization
OptsSolver.print_time = printTime;

% Disable solver output
switch solver
  case 'qpoases'
    OptsSolver.printLevel = 'low';
    Opti.solver( 'qpoases', OptsSolver );
    
  case 'osqp'
    Opti.solver( 'osqp', OptsSolver );
    
  case 'hpipm'
    Opti.solver( 'hpipm', OptsSolver );
    
  case 'qrqp'
    OptsSolver.print_iter = 0;
    Opti.solver( 'qrqp', OptsSolver );
    
  case 'ipopt'
    OptsSolver.ipopt.print_level = 1;
    Opti.solver( 'ipopt', OptsSolver );
end

% Store Opti instance and variables that need to be updated or evaluated at
% solution in output struct
Solver.Opti = Opti;
Solver.x_ = x_;
Solver.u_ = u_;
Solver.xk_ = xk_;
Solver.xSafeSet_ = xSafeSet_;
Solver.Q_ = Q_ ;

end

           
function [X, U, costToGo] = sim_lmpc( Solver, x0, dataX, dataQ, A, B, Q, R, nIter, tol )
%
%   Signature   : [X, U, costToGo] = sim_lmpc( Solver, x0, dataX, dataQ, A, B, Q, R, nIter, tol )
%
%   Description : Learning Model Predictive Control simulation loop. This
%                 function runs the LMPC controller for a user-defined 
%                 number of iterations.
% 
%   Parameters  : Solver -> Predefined solver object
%                 x0 -> Initial state for each iteration
%                 dataX, dataQ -> Initial sampled safe set and
%                                 corresponding cost-to-go
%                 A, B -> Linear state space matrices
%                 Q, R -> Weighting matrices for the cost function
%                 nIter -> Number of iterations to be simulated
%                 tol -> Numerical tolerance for which origin is accepted 
%                        to be reached ||x||_2 < tol
% 
%   Return      : X -> Cell array containing the state trajectories of
%                      each iteration
%                 U -> Cell array containing the input trajectories of
%                      each iteration
%                 costToGo -> Cell array containing the cost-to-go of each
%                             iteration
% 
%-------------------------------------------------------------------------%

% Initialize arrays for iteration cost, state and input trajectory for each
% iteration
costToGo = cell(nIter,1);
X = cell(nIter,1);
U = cell(nIter,1);
  
fprintf('Iteration j= 0: Cumulative cost = %8.5f (initial trajectory)\n', dataQ(1));

% Run LMPC for selected number of iterations
for jj = 1:nIter

  % Reset exit flag
  exitFlag = 0;
  
  % Initialize time index
  kk = 1;
  
  % Reset state and input trajectory
  xLmpc = x0;
  uLmpc = 0;
  
  % Update ocp parameters for current iteration jj
  Solver.Opti.set_value( Solver.xSafeSet_, dataX );
  Solver.Opti.set_value( Solver.Q_, dataQ );
  
  % Simulate one iteration
  while exitFlag == 0

    % Update ocp parameters for current time step tt
    Solver.Opti.set_value( Solver.xk_, xLmpc(:,kk) );
    
    % Solve the FTOCP at time tt of the jj-th iteration
    sol = Solver.Opti.solve();
    
    % Simulate one step of the environment/ dynamic system
    uLmpc(:,kk) = full( sol.value( Solver.u_(:,1)) );
    xLmpc(:,kk+1) = A*xLmpc(:,kk) + B*uLmpc(:,kk) ;

    % Check exit conditions
    if xLmpc(:,kk+1)'*xLmpc(:,kk+1) < tol
      exitFlag = 1;
    end
    
    % Increment time index
    kk = kk + 1;
  end
  
  % Compute iteration cost of finalized iteration
  costIterJ = compute_cost(xLmpc, uLmpc, Q, R );
  
  % Append new trajectory, iteration cost and safe set to storage arrays
  X{jj} = xLmpc;
  U{jj} = uLmpc;
  costToGo{jj} = costIterJ;
  
  % Update safe set and Q function
  for ii = 1:size(xLmpc,2)
    if ~any( vecnorm( dataX - xLmpc(:,ii), 2) < 1e-09 )
      dataX(:,ii) = xLmpc(:,ii);
      dataQ(:,ii) = costIterJ(ii);
    end
  end

  fprintf('Iteration j=%2d: Cumulative cost = %8.5f\n', [jj, costToGo{jj}(1)]);
end
end


function [X, U] = compute_optimal_trajectory( x0, N, Q, R, A, B, xMax, xMin, uMax, uMin )
% 
%   Signature   : [X, U] = compute_optimal_trajectory( x0, N, Q, R, A, B, xMax, xMin, uMax, uMin )
%
%   Description : Solve the OCP with the given parameters.
%
%   Parameters  : x0 -> Initial state
%                 N -> Prediction horizon
%                 Q, R -> Weighting matrices for the cost function
%                 A, B -> Linear state space matrices
%                 xMax, xMin, uMax, uMin -> Bounds on states and inputs
% 
%   Return      : X, U -> Optimal state and input trajectory
%
%-------------------------------------------------------------------------%
            
Ocp0 = casadi.Opti();
x_ = Ocp0.variable( size(A,1), N+1 );
u_ = Ocp0.variable( size(B,2), N );

%% Constraints
% Initial condition
Ocp0.subject_to( x_(:,1) == x0 );

% System dynamics
Ocp0.subject_to( x_(:,2:end) == A*x_(:,1:end-1) + B*u_ );

% Bounds
Ocp0.subject_to( xMin <= x_(:,1:end-1) <= xMax );
Ocp0.subject_to( uMin <= u_ <= uMax );

% Terminal constraint
Ocp0.subject_to( x_(:,end) == [0;0] );

%% Cost
% Stage cost
cost = [];
for ii = 1:N
    cost = [cost; x_(:,ii).' * Q * x_(:,ii) + u_(:,ii).' * R * u_(:,ii) ];
end

% Terminal cost
cost = [cost; x_(:,end).' * Q * x_(:,end) ];

% Solve the finite time optimal control problem (FTOCP)
Ocp0.minimize( sum(cost) );
Ocp0.solver( 'ipopt', struct('expand',1, 'print_time',0, 'ipopt',struct('print_level',0) ) );
sol = Ocp0.solve();

% Store numerical solution
X = sol.value( x_ );
U = sol.value( u_ );

end


function Q = compute_cost( x, u, Q, R )
% 
%   Signature   : Q = compute_cost( x, u, Q, R )
%
%   Description : Compute cost-to-go Q for each point in the trajectory
%                 defined by x and u.
%
%   Parameters  : x, u -> State and input trajectory
%                 Q, R -> Weighting matrices for the cost function
% 
%   Return      : Q -> Cost-to-go
%
%-------------------------------------------------------------------------%

cost = zeros(1, size(x,2));
cost(1) = x(:,end).' * Q * x(:,end);

for ii = 2:(size(x,2))
  cost(ii) = cost(ii-1) + x(:,end-ii+1).' * Q * x(:,end-ii+1) + ...
             u(:,end-ii+2).' * R * u(:,end-ii+2);
end

Q = flip(cost);

end


function plot_results( x0, u0, Q0, X, U, Q, xOpt, uOpt, QOpt, nIter )
% 
%   Signature   : plot_results( x0, u0, Q0, X, U, Q, xOpt, uOpt, QOpt, nIter )
%
%   Description : Plot simulation results.
%
%   Parameters  : -
% 
%   Return      : -
%
%-------------------------------------------------------------------------%

figure( 'Name', 'Evolution over iterations' )
subplot(131); grid on; hold on; xlabel( 'x_1' ); ylabel( 'x_2' )
xlim([-16,2])
p111 = plot( x0(1,:), x0(2,:), 'marker', 'D' );
for ii = 1:length(X)-1
  plot( X{ii}(1,:), X{ii}(2,:), '-x' )
end
p112 = plot( X{end}(1,:), X{end}(2,:), 'k', 'marker', '.', 'markersize', 10 );
legend( [p111, p112], 'x1-x2, j=0', ['x1-x2, j=', num2str(nIter)] )

subplot(132); grid on; hold on; xlabel( 'Time index k' ); ylabel( 'x1/x2' )
xlim([1,16])
p121 = plot( x0(1,:), 'marker', 'D' );
p122 = plot( x0(2,:), 'marker', 'D' );
for ii = 1:length(X)
  plot( X{ii}(1,:), '-x' )
  plot( X{ii}(2,:), '-x' )
end
p123 = plot( X{ii}(1,:), 'k', 'marker', '.', 'markersize', 10 );
p124 = plot( X{ii}(2,:), 'color', [0 84 159]/255, 'marker', '.', 'markersize', 10 );
legend( [p121, p122, p123, p124], 'x1, j=0', 'x2, j=0', ...
        ['x1, j=', num2str(nIter)], ['x2, j=', num2str(nIter)] )

subplot(133); grid on; hold on; xlabel( 'Time index k' ); ylabel( 'u' )
xlim([1,16])
p131 = plot( u0, 'marker', 'D' );
for ii = 1:length(U)-1
  plot( U{ii}, '-x' )
end
p132 = plot( U{end}, 'k', 'marker', '.', 'markersize', 10 );
legend( [p131, p132], 'u, j=0', ['u, j=', num2str(nIter)])


figure( 'Name', 'Comparison to optimal solution' )
subplot(131); grid on; hold on; xlabel( 'x_1' ); ylabel( 'x_2' )
xlim([-16,2])
plot( xOpt(1,:), xOpt(2,:), 'marker', 'D' )
plot( X{end}(1,:), X{end}(2,:), 'k', 'marker', '.', 'markersize', 10 )
legend( 'x1-x2, opt', ['x1-x2, j=', num2str(nIter)] )

subplot(132); grid on; hold on; xlabel( 'Time index k' ); ylabel( 'x_2' )
xlim( [1,16] )
p221 = plot( xOpt(1,:), 'marker', 'D' );
p222 = plot( xOpt(2,:), 'marker', 'D' );
p223 = plot( X{end}(1,:), 'k', 'marker', '.', 'markersize', 10 );
p224 = plot( X{end}(2,:), 'color', [0 84 159]/255, 'marker', '.', 'markersize', 10 );
legend( [p221, p222, p223, p224], 'x1, opt', 'x2, opt', ['x1, j=', num2str(nIter)], ['x2, j=', num2str(nIter)] )

subplot(133); grid on; hold on; xlabel( 'Time index k' ); ylabel( 'u' )
xlim( [1,16] )
plot( uOpt, 'marker', 'd' );
plot( U{end}, 'k', 'marker', '.', 'markersize', 10 )
legend( 'u, opt', ['u, j=', num2str(nIter)] )


figure( 'Name', 'Overall cost over iterations' )
grid on; hold on; ylim([450,600]); xlabel('Iteration j'); ylabel('J(x_0)')
Q = cell2mat(Q);
plot( [Q0(1); Q(:,1)], 'marker', '.', 'markersize', 10 )
plot( 1:nIter+1, QOpt(1)*ones(nIter+1,1), 'r' )
legend( 'LMPC', 'Opt' )
annotation( 'textbox', [7/11 50/150 .1 .1], 'String', ...
            ['Delta cost: ', num2str(Q(end,1)-QOpt(1), '%2.2e')], ...
            'BackgroundColor', 'r', 'FaceAlpha', 0.5 )

end