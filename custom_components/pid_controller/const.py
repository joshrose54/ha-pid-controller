"""
Constants for the PID Controller component.
This module defines constants used throughout the PID controller implementation.

For more details about this sensor, please refer to the documentation at https://github.com/joshrose54/ha-pid-controller
"""

# Component details
COMPONENT_DOMAIN = "pid_controller"
VERSION = "1.0.0"

# Services provided by this component
COMPONENT_SERVICES = "pid-services"
SERVICE_RESET_PID = "reset_pid"
SERVICE_AUTOTUNE = "autotune_pid"

# Configuration keys
CONF_SENSOR = "sensor_entity"
CONF_SETPOINT = "set_point"
CONF_PROPORTIONAL = "p"
CONF_INTEGRAL = "i"
CONF_DERIVATIVE = "d"
CONF_INVERT = "invert"
CONF_PRECISION = "precision"
CONF_ROUND = "round"
CONF_SAMPLE_TIME = "sample_time"
CONF_WINDUP = "windup"
CONF_ENABLED = "enabled"
CONF_SHUT_DOWN_HIGH = "shut_down_high"

# Default values
DEFAULT_NAME = "PID Controller"
DEFAULT_PRECISION = 2
DEFAULT_MINIMUM = 0
DEFAULT_MAXIMUM = 1
DEFAULT_ROUND = "round"
DEFAULT_SAMPLE_TIME = 0
DEFAULT_WINDUP = 20
DEFAULT_UNIT_OF_MEASUREMENT = "points"
DEFAULT_DEVICE_CLASS = "None"
DEFAULT_ICON = "mdi:chart-bell-curve-cumulative"
DEFAULT_ENABLED = True
DEFAULT_SHUT_DOWN_HIGH = 23

# Other
ROUND_FLOOR = "floor"
ROUND_CEIL = "ceil"
ROUND_ROUND = "round"

# Attributes
ATTR_ENABLED = "enabled"
ATTR_TUNNING = "tunning"
ATTR_ICON = "icon"
ATTR_UNIT_OF_MEASUREMENT = "unit_of_measuremnt"
ATTR_DEVICE_CLASS = "device_class"
ATTR_PROPORTIONAL = "proportional"
ATTR_INTEGRAL = "integral"
ATTR_DERIVATIVE = "derivative"
ATTR_SENSOR_ENTITY = "sensor_entity"
ATTR_SETPOINT = "set_point"
ATTR_SOURCE = "source"
ATTR_PRECISION = "precision"
ATTR_MINIMUM = "minimum"
ATTR_MAXIMUM = "maximum"
ATTR_RAW_STATE = "raw_state"
ATTR_SAMPLE_TIME = "sample_time"
ATTR_INVERT = "invert"
ATTR_P = "p"
ATTR_I = "i"
ATTR_D = "d"
ATTR_SHUT_DOWN_HIGH = "shut_down_high"

# Mapping of attributes to internal properties
ATTR_TO_PROPERTY = [
    ATTR_ENABLED,
    ATTR_TUNNING,
    ATTR_ICON,
    ATTR_UNIT_OF_MEASUREMENT,
    ATTR_DEVICE_CLASS,
    ATTR_PROPORTIONAL,
    ATTR_INTEGRAL,
    ATTR_DERIVATIVE,
    ATTR_SENSOR_ENTITY,
    ATTR_SETPOINT,
    ATTR_SOURCE,
    ATTR_PRECISION,
    ATTR_MINIMUM,
    ATTR_MAXIMUM,
    ATTR_RAW_STATE,
    ATTR_INVERT,
    ATTR_SAMPLE_TIME,
    ATTR_P,
    ATTR_I,
    ATTR_D,
    ATTR_SHUT_DOWN_HIGH,
]
