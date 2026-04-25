classdef StraightLineEnv < rl.env.MATLABEnvironment

    properties
        % Simulation parameters
        aircraft_parameters
        aircraft_state_trim
        control_input_trim
        trim_variables
        
        % Guidance line
        pos_line = [0; 0; -1805]
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
        prev_obs = zeros(4,1)   % last raw 4-state obs — used for finite-difference rates
        difficulty = 0          % curriculum stage [0=easy, 1=hard]; set from TrainPPO.m
    end

    properties (Access = protected)
        IsDone = false
    end

    methods
        function this = StraightLineEnv()
            % --- Define observation space ---
            % [cross_track, course_err, h_err, Va_err,
            %  d_cross_track/dt (m/s), d_course_err/dt (rad/s)]
            % The two rate terms give the policy derivative information so it
            % can damp its own corrections rather than oscillating.
            % Obs are normalized to [-1, 1] in step() and reset() before
            % being returned to the agent, so bounds are ±1 here.
            obsInfo = rlNumericSpec([6 1], ...
                'LowerLimit', -ones(6,1), ...
                'UpperLimit',  ones(6,1));
            obsInfo.Name = 'observations';

            % --- Define action space ---
            % PPO outputs DELTA corrections: [dchi (rad), dh (m), dVa (m/s)]
            % Applied on top of baseline (chi_des, h_des, V_trim) each step
            actInfo = rlNumericSpec([3 1], ...
                'LowerLimit', [-pi/6; -30; -3], ...
                'UpperLimit', [ pi/6;  30;  3]);
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
            load('ttwistor_gains_feed', '-mat');
            this.control_gain_struct = control_gain_struct;  % adjust to match your gains struct variable name
            this.control_gain_struct.Ts = this.Ts;
            this.control_gain_struct.takeoff_height = -999;     % aircraft never below this → takeoff never triggers
            this.control_gain_struct.height_hold_limit = 99999; % band is enormous → always in altitude hold

        end

        function [obs, reward, isDone, info] = step(this, action)
            i = this.StepCount;
            TSPAN = this.Ts * [i, i+1];

            % --- Unpack action (RL policy: deltas from baseline) ---
            chi_des = atan2(this.dir_line(2), this.dir_line(1));  % desired heading
            chi_cmd = chi_des + action(1);           % baseline heading + correction
            h_cmd   = -this.pos_line(3) + action(2); % desired altitude + correction
            Va_cmd  = this.V_trim + action(3);        % trim airspeed + correction

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
            [control_out, ~] = hw7utils.SLCWithFeedForwardAutopilot( ...
                this.Ts * i, this.aircraft_state, wind_angles, ...
                control_objectives, this.control_gain_struct);

            % --- Propagate dynamics ---
            [~, YOUT] = ode45(@(t,y) utils.AircraftEOM( ...
                t, y, control_out, this.wind_inertial, this.aircraft_parameters), ...
                TSPAN, this.aircraft_state, []);

            this.aircraft_state = YOUT(end,:)';
            this.StepCount = this.StepCount + 1;

            % --- Compute raw 4-state observation ---
            raw_obs = this.computeObservation();

            % --- Finite-difference rate terms ---
            d_cross  = (raw_obs(1) - this.prev_obs(1)) / this.Ts;  % cross-track rate (m/s)
            d_course = (raw_obs(2) - this.prev_obs(2)) / this.Ts;  % course rate     (rad/s)
            this.prev_obs = raw_obs;
            % Normalize to [-1, 1] — keeps all inputs O(1) for the network
            obs = [raw_obs(1)/500; raw_obs(2)/pi; raw_obs(3)/200; raw_obs(4)/10; ...
                   d_cross/20;    d_course/0.5];

            % --- Compute reward (uses base 4 states + cross-track rate) ---
            reward = this.computeReward(raw_obs, d_cross);

            % --- Check termination ---
            % Termination at 400 m: wide enough for 3σ of difficulty=1 ICs
            % (σ_pos=135m → 3σ=405m). Keeps learning signal for hard resets
            % rather than immediately terminating.
            isDone = this.StepCount >= this.MaxSteps || abs(raw_obs(1)) > 400; % 200 before
            this.IsDone = isDone;
            info = [];
        end

        function obs = reset(this)
            this.aircraft_state = this.aircraft_state_trim;

            % Curriculum-scaled ICs — controlled via env.difficulty in TrainPPO.m
            %   difficulty=0.00 : sig_pos=15m,  sig_chi=0.15 rad,  sig_h= 5m
            %   difficulty=0.33 : sig_pos=55m,  sig_chi=0.23 rad,  sig_h=11m
            %   difficulty=0.67 : sig_pos=95m,  sig_chi=0.32 rad,  sig_h=18m
            %   difficulty=1.00 : sig_pos=135m, sig_chi=0.40 rad,  sig_h=25m
            sig_pos = 15  + 120  * this.difficulty;
            sig_chi = 0.15 + 0.25 * this.difficulty;
            sig_h   = 5   + 20   * this.difficulty;   % altitude IC spread (m)

            this.aircraft_state(1) = this.aircraft_state_trim(1) + randn()*sig_pos;
            this.aircraft_state(2) = this.aircraft_state_trim(2) + randn()*sig_pos;
            % Altitude perturbation: state(3) is NED-down, so subtract to increase altitude
            this.aircraft_state(3) = this.aircraft_state_trim(3) - randn()*sig_h;

            chi_des = atan2(this.dir_line(2), this.dir_line(1));
            this.aircraft_state(6) = chi_des + randn()*sig_chi;

            this.StepCount = 0;
            this.IsDone = false;
            raw_obs = this.computeObservation();
            this.prev_obs = raw_obs;   % seed so first step rate = 0
            % Return normalized obs with zero initial rates
            obs = [raw_obs(1)/500; raw_obs(2)/pi; raw_obs(3)/200; raw_obs(4)/10; 0; 0];
            fprintf('Reset: cross_track=%.1f m | course_err=%.2f rad | h_err=%.1f m  [difficulty=%.2f]\n', ...
                raw_obs(1), raw_obs(2), raw_obs(3), this.difficulty);
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

        function reward = computeReward(this, obs, d_cross)
            cross_track = obs(1);
            course_err  = obs(2);
            h_err       = obs(3);
            Va_err      = obs(4);
            ct = abs(cross_track);

            % ── CROSS-TRACK PENALTY ────────────────────────────────────────────────
            % Piecewise: quadratic within 50 m (tight precision gradient),
            % then linear continuation so gradient stays non-zero out to 200 m
            % instead of saturating near zero (old pure-exponential behaviour).
            if ct <= 50
                r_cross = 1.0 * (ct / 50)^2;                        % 0 → 1 within 50 m
            else
                r_cross = 1.0 * min(1.0, 0.5 + (ct - 50) / 300);   % linear, caps at 1.0
            end

            % ── GUIDANCE REWARD (blended far/near) ────────────────────────────────
            % The agent is free to discover its own approach law.
            % We do NOT encode the SLC’s atan approach here — instead, we
            % reward the OUTCOME (convergence) far from the line and
            % reward PRECISION (course alignment) near the line.
            %
            % blend = 1  near the line (≤ ~35 m) → course alignment matters
            % blend = 0  far from line           → closing speed matters
            blend = exp(-ct^2 / 1250);   % σ ≈ 35 m

            % Closing speed: positive when moving toward line, negative when diverging
            closing = -sign(cross_track) * d_cross;   % m/s toward line
            r_closing = (1 - blend) * 0.5 * tanh(closing / 5);
            % +0.5 when converging at 5 m/s, -0.5 when diverging at 5 m/s

            % Course alignment: only counts near the line
            r_course = -blend * 0.5 * (1 - exp(-course_err^2 / 0.3));
            % 0 when on the line with correct course, -0.5 at large course_err

            % ── ALTITUDE & AIRSPEED ───────────────────────────────────────────────
            r_h  = 0.2 * (1 - exp(-h_err^2  / 2500));
            r_va = 0.1 * (1 - exp(-Va_err^2 / 9));

            % ── TOTAL ───────────────────────────────────────────────────────
            % Max per step: 1.0 - 0 + 0 + 0 - 0 - 0 + 1.0 = 2.0  (on line, converging)
            % Floor (200m, diverging at 5 m/s): 1.0 - 1.0 - 0.5 - 0 - 0.2 - 0.1 = -0.8
            reward = 1.0 - r_cross + r_closing + r_course - r_h - r_va;

            % Two-tier proximity bonus — incentivises pushing all the way to 1 m,
            % not just loitering at 4.9 m. Max per-step is still 2.0.
            %   |ct| < 5 m : +0.25  (approach zone — partial credit)
            %   |ct| < 1 m : +1.00  (tight tracking — full bonus; 1.25 total)
            if ct < 1.0
                reward = reward + 1.0;    % full tight-tracking bonus
            elseif ct < 5.0
                reward = reward + 0.25;   % partial approach-zone bonus
            end
        end
    end
end