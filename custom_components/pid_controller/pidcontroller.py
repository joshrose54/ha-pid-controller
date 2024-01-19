
"""
PID Controller Implementation.
This module contains the logic for a Proportional-Integral-Derivative (PID) controller.

For more details about this sensor, please refer to the documentation at https://github.com/joshrose54/ha-pid-controller
"""

import time

class PIDController:
    """
    Class representing a PID Controller.
    """

    WARMUP_STAGE = 3

    def __init__(self, P=0.2, I=0.0, D=0.0, logger=None):
        """
        Initialize the PID controller with given PID constants.
        """
        
        self._logger = logger

        self._set_point = 0
        self._windup = (None, None)
        self._output = 0.0

        self._kp = P
        self._ki = I
        self._kd = D

        self._p_term = 0.0
        self._i_term = 0.0
        self._d_term = 0.0

        self._sample_time = None
        self._last_output = None
        self._last_input = None
        self._last_time = None

        self.reset_pid()

    def reset_pid(self):
        """
        Reset the PID controller terms.
        """
        
        self._p_term = 0.0
        self._i_term = 0.0
        self._d_term = 0.0

        self._sample_time = None
        self._last_output = None
        self._last_input = None
        self._last_time = None

    def update(self, feedback_value, in_time=None):
        """
        Update the PID controller with a new feedback value.
        """

        current_time = in_time if in_time is not None else self.current_time()
        if self._last_time is None:
            self._last_time = current_time

        # Fill PID information
        delta_time = current_time - self._last_time
        if not delta_time:
            delta_time = 1e-16
        elif delta_time < 0:
            return

        # Return last output if sample time not met
        if (
            self._sample_time is not None
            and self._last_output is not None
            and delta_time < self._sample_time
        ):
            return self._last_output

        # Calculate error
        error = self._set_point - feedback_value
        last_error = self._set_point - (
            self._last_input if self._last_input is not None else self._set_point
        )

        # Calculate delta error
        delta_error = error - last_error
        
        # Calculate delta input
        delta_feedback_value = feedback_value - ( self._last_input if self._last_input is not None else 0)

        # Calculate P
        self._p_term = self._kp * error

        # Calculate I
        i_term_delta = self._ki * error
        i_term_delta = self.clamp_value(i_term_delta, self._windup)
        self._i_term += i_term_delta
        self._i_term = self.clamp_value(self._i_term, (0, 100))

        # Calculate D
        self._d_term = self._kd * delta_feedback_value

        # Compute final output
        self._output = self._p_term + self._i_term + self._d_term
        self._output = self.clamp_value(self._output, (0, 100))

        # Keep Track
        self._last_output = self._output
        self._last_input = feedback_value
        self._last_time = current_time

    @property
    def kp(self):
        """
        Get the Proportional Gain (Kp) of the PID controller.
        This value determines how aggressively the PID reacts to the current error.
        """
        return self._kp

    @kp.setter
    def kp(self, value):
        """
        Set the Proportional Gain (Kp) of the PID controller.
        Adjusting this value changes how aggressively the PID responds to the current error.
        """
        self._kp = value

    @property
    def ki(self):
       """
        Get the Integral Gain (Ki) of the PID controller.
        This value determines how aggressively the PID reacts to the accumulated error over time.
        """
        return self._ki

    @ki.setter
    def ki(self, value):
        """
        Set the Integral Gain (Ki) of the PID controller.
        Adjusting this value changes how the PID responds to the accumulated error over time.
        """
        self._ki = value

    @property
    def kd(self):
        """
        Get the Derivative Gain (Kd) of the PID controller.
        This value determines how aggressively the PID reacts to the rate of change of the error.
        """
        return self._kd

    @kd.setter
    def kd(self, value):
        """
        Set the Derivative Gain (Kd) of the PID controller.
        Adjusting this value changes how the PID responds to the rate of change of the error.
        """
        self._kd = value

    @property
    def set_point(self):
        """
        Get the target set point of the PID controller.
        The PID controller will attempt to adjust the process to meet this set point.
        """
        return self._set_point

    @set_point.setter
    def set_point(self, value):
        """
        Set the target set point of the PID controller.
        Adjusting this value changes the target that the PID controller will try to achieve.
        """
        self._set_point = value

    @property
    def windup(self):
        """
        Get the windup limit for the integral term of the PID controller.
        This value prevents the integral term from accumulating excessive error.
        """
        return self._windup

    @windup.setter
    
    def windup(self, value):
        """
        Set the windup limit for the integral term of the PID controller.
        Adjusting this value sets a limit to prevent excessive accumulation of the integral error.
        """
        self._windup = (-value, value)

    @property
    def sample_time(self):
        """
        Get the sample time for the PID controller.
        This is the interval at which the PID controller updates its calculations.
        """
        return self._sample_time

    @sample_time.setter
    def sample_time(self, value):
        """
        Set the sample time for the PID controller.
        Adjusting this value changes the frequency at which the PID controller updates its output.
        """
        self._sample_time = value

    @property
    def p(self):
        return self._p_term

    @property
    def i(self):
        return self._i_term

    @property
    def d(self):
        return self._d_term

    @property
    def output(self):
        """
        Get the current output of the PID controller.
        This is the computed control variable that should be applied to the process.
        """
        return self._output

    def log(self, message):
        """
        Log a message using the controller's logger.
        """
        
        if not self._logger:
            return
        self._logger.warning(message)

    def current_time(self):
        """
        Get the current time, using monotonic time if available.
        """
        
        try:
            ret_time = time.monotonic()
        except AttributeError:
            ret_time = time.time()

        return ret_time

    def clamp_value(self, value, limits):
        """
        Clamp a value within specified limits.
        """
        
        lower, upper = limits

        if value is None:
            return None
        elif not lower and not upper:
            return value
        elif (upper is not None) and (value > upper):
            return upper
        elif (lower is not None) and (value < lower):
            return lower
        return value
