classdef StraightLineEnv < rl.env.MATLABEnvironment

    properties
        % Simulation parameters
        aircraft_parameters
        aircraft_state_trim
        control_input_trim
        trim_variables
        
        % Guidance line
        pos_line = [300; 0; -1805]
        dir_line  % normalized
        kpath = 0.01
        chi_inf   % max course error (rad)
        V_trim = 18
        
        % Control gains struct (for autopilot)
        control_gain_struct
        
        % Sim settings
        Ts = 0.1
        MaxSteps = 3000   % 300s / 0.1
        
        % Current state
        aircraft_state
        wind_inertial = [0;0;0]
        StepCount = 0
    end

    properties (Access = protected)
        IsDone = false
    end

    methods
        function this = StraightLineEnv()
            % --- Define observation space ---
            % [cross-track error, course error, height error, airspeed error]
            obsInfo = rlNumericSpec([4 1], ...
                'LowerLimit', [-500; -pi; -200; -10], ...
                'UpperLimit', [ 500;  pi;  200;  10]);
            obsInfo.Name = 'observations';

            % --- Define action space ---
            % PPO outputs [chi_cmd (rad), h_cmd (m), Va_cmd (m/s)]
            % Or alternatively: corrections to the baseline guidance output
            actInfo = rlNumericSpec([3 1], ...
                'LowerLimit', [-pi/4; 1600; 14], ...
                'UpperLimit', [ pi/4; 2000; 22]);
            actInfo.Name = 'actions';

            this = this@rl.env.MATLABEnvironment(obsInfo, actInfo);
            this.initialize();
        end

        function initialize(this)
            % Load aircraft model (mirrors your RunHW7 setup)
            this.aircraft_parameters = utils.ttwistor();
            
            trim_def = [this.V_trim; 0; 1805];
            [this.aircraft_state_trim, this.control_input_trim, ...
             this.trim_variables, ~] = utils.CalculateTrim(trim_def, this.aircraft_parameters);

            d = [sqrt(2)/2; 8*sqrt(2)/2; 0];
            this.dir_line = d / norm(d);
            this.chi_inf = 10 * pi/180;

            % Load gains
            load('ttwistor_gains_slc', '-mat');
            this.control_gain_struct = gains;  % adjust to match your gains struct variable name
            this.control_gain_struct.Ts = this.Ts;
        end

        function [obs, reward, isDone, info] = step(this, action)
            i = this.StepCount;
            TSPAN = this.Ts * [i, i+1];

            % --- Unpack action (RL policy output) ---
            chi_cmd = action(1);
            h_cmd   = action(2);
            Va_cmd  = action(3);

            % Build control_objectives from RL action
            % Format expected by your autopilot:
            % [h_cmd; gamma_cmd; chi_cmd; chi_dot_cmd; Va_cmd]
            control_objectives = [h_cmd; 0; chi_cmd; 0; Va_cmd];

            % --- Compute wind angles ---
            wind_body = utils.TransformFromInertialToBody( ...
                this.wind_inertial, this.aircraft_state(4:6));
            air_rel = this.aircraft_state(7:9) - wind_body;
            wind_angles = utils.AirRelativeVelocityVectorToWindAngles(air_rel);

            % --- Run autopilot (SLC inner loop) ---
            [control_out, ~] = frewhw6utils.SimpleSLCAutopilot( ...
                this.Ts * i, this.aircraft_state, wind_angles, ...
                control_objectives, this.control_gain_struct);

            % --- Propagate dynamics ---
            [~, YOUT] = ode45(@(t,y) utils.AircraftEOM( ...
                t, y, control_out, this.wind_inertial, this.aircraft_parameters), ...
                TSPAN, this.aircraft_state, []);

            this.aircraft_state = YOUT(end,:)';
            this.StepCount = this.StepCount + 1;

            % --- Compute observation ---
            obs = this.computeObservation();

            % --- Compute reward ---
            reward = this.computeReward(obs);

            % --- Check termination ---
            isDone = this.StepCount >= this.MaxSteps || abs(obs(1)) > 400;
            this.IsDone = isDone;
            info = [];
        end

        function obs = reset(this)
            this.aircraft_state = this.aircraft_state_trim;
            % Randomize initial position slightly for robustness
            this.aircraft_state(1) = this.aircraft_state_trim(1) + randn()*20;
            this.aircraft_state(2) = this.aircraft_state_trim(2) + randn()*20;
            this.StepCount = 0;
            this.IsDone = false;
            obs = this.computeObservation();
        end
    end

    methods (Access = private)
        function obs = computeObservation(this)
            p = this.aircraft_state(1:3);
            h_actual = -p(3);

            % Cross-track error (signed distance from line)
            dp = p - this.pos_line;
            cross_track = norm(dp - dot(dp, this.dir_line)*this.dir_line);
            % Sign: positive = left of line
            side = cross(this.dir_line, dp/norm(dp));
            cross_track = sign(side(3)) * cross_track;

            % Course error
            u = this.aircraft_state(7); v = this.aircraft_state(8);
            psi = this.aircraft_state(6);
            Vn = u*cos(psi) - v*sin(psi);
            Ve = u*sin(psi) + v*cos(psi);
            chi_actual = atan2(Ve, Vn);
            chi_des = atan2(this.dir_line(2), this.dir_line(1));
            course_err = wrapToPi(chi_actual - chi_des);

            % Height error
            h_des = -this.pos_line(3);
            h_err = h_actual - h_des;

            % Airspeed error
            Va_actual = norm(this.aircraft_state(7:9));
            Va_err = Va_actual - this.V_trim;

            obs = [cross_track; course_err; h_err; Va_err];
        end

        function reward = computeReward(this, obs)
            cross_track = obs(1);
            course_err  = obs(2);
            h_err       = obs(3);
            Va_err      = obs(4);

            % Dense reward: penalize errors
            reward = - 0.01  * cross_track^2 ...
                     - 1.0   * course_err^2  ...
                     - 0.005 * h_err^2       ...
                     - 0.1   * Va_err^2;

            % Bonus for staying close to line
            if abs(cross_track) < 5
                reward = reward + 1.0;
            end

            % Hard crash penalty
            if abs(cross_track) > 400
                reward = reward - 100;
            end
        end
    end
end