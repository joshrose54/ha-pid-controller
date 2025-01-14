"""
PID Controller Component Initialization.
This module sets up the PID controller component and its services in Home Assistant.

For more details about this sensor, please refer to the documentation at https://github.com/joshrose54/ha-pid-controller
"""
import logging
from distutils import util

import homeassistant.helpers.config_validation as cv
from homeassistant.core import HomeAssistant
from homeassistant.helpers.service import verify_domain_control
from homeassistant.const import ATTR_ENTITY_ID
from homeassistant.exceptions import HomeAssistantError
import voluptuous as vol
from .const import *

__version__ = VERSION

_LOGGER = logging.getLogger(__name__)

# Define the schema for the services provided by this component
SERVICE_SCHEMA = vol.Schema(
    {
        vol.Required(ATTR_ENTITY_ID): cv.entity_id,
    }
)

# pylint: disable=unused-argument
async def async_setup(hass: HomeAssistant, config):
    """
    Async setup function to initialize the PID controller component.
    Registers the PID reset and autotune services.
    """
    
    _LOGGER.debug("setup")

    async def async_pid_service_reset(call) -> None:
        """Call pid service handler."""
        _LOGGER.info("%s service called", call.service)
        await pid_reset_service(hass, call)

    hass.services.async_register(
        COMPONENT_DOMAIN,
        SERVICE_RESET_PID,
        async_pid_service_reset,
        schema=SERVICE_SCHEMA,
    )

    async def async_pid_service_autotune(call) -> None:
        """Call pid service handler."""
        _LOGGER.info("%s service called", call.service)
        await pid_autotune_service(hass, call)

    hass.services.async_register(
        COMPONENT_DOMAIN,
        SERVICE_AUTOTUNE,
        async_pid_service_autotune,
        schema=SERVICE_SCHEMA,
    )

    return True


def get_entity_from_domain(hass: HomeAssistant, domain, entity_id):
    """
    Helper function to retrieve an entity from a domain.
    Raises an error if the domain or entity is not found.
    """
    
    component = hass.data.get(domain)
    if component is None:
        raise HomeAssistantError(f"{domain} component not set up")

    entity = component.get_entity(entity_id)
    if entity is None:
        raise HomeAssistantError(f"{entity_id} not found")

    return entity


async def pid_reset_service(hass: HomeAssistant, call):
    """
    Service handler for resetting a PID controller.
    """
    
    entity_id = call.data["entity_id"]
    domain = entity_id.split(".")[0]

    _LOGGER.info("%s reset pid", entity_id)

    try:
        get_entity_from_domain(hass, domain, entity_id).reset_pid()
    except AttributeError:
        raise HomeAssistantError(f"{entity_id} can't reset PID") from AttributeError


async def pid_autotune_service(hass: HomeAssistant, call):
    """
    Service handler for autotuning a PID controller.
    """
    
    entity_id = call.data["entity_id"]
    domain = entity_id.split(".")[0]

    _LOGGER.info("%s autotune pid", entity_id)

    try:
        get_entity_from_domain(hass, domain, entity_id).start_autotune()
    except AttributeError:
        raise HomeAssistantError(f"{entity_id} can't reset PID") from AttributeError
