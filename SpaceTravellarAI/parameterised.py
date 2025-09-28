import re


class ParameterisedCommand:
    """ Handle commands that require specific numeric or categorical parameters. """

    def __init__(self):
        # Define the range for different command types
        self.parameter_ranges = {
            'adjust_trajectory': {
                'angle_x': (-45.0, 45.0),  # Degrees
                'angle_y': (-45.0, 45.0),  # Degrees
                'angle_z': (-45.0, 45.0),  # Degrees
                'magnitude': (0.1, 1.0)  # Relative thrust magnitude
            },
            'increase_velocity': {
                'delta_v': (0.1, 5.0)  # Speed increase in units/second
            },
            'decrease_velocity': {
                'delta_v': (0.1, 5.0)  # Speed decrease in units/second
            },
            'investigate_anomaly': {
                'scan_intensity': (1, 10),  # Level of scan detail
                'approach_distance': (10, 100)  # How close to get (meters)
            },
            'refuel': {
                'amount': (0.1, 1.0)  # Relative amount (percentage of max)
            }
        }
        # Default values for when parameters are not specified
        self.default_values = {
            'adjust_trajectory': {'angle_x': 0.0, 'angle_y': 0.0, 'angle_z': 0.0, 'magnitude': 0.5},
            'increase_velocity': {'delta_v': 1.0},
            'decrease_velocity': {'delta_v': 1.0},
            'investigate_anomaly': {'scan_intensity': 5, 'approach_distance': 50},
            'refuel': {'amount': 1.0}
        }

    def create_command(self, command_type, **parameters):
        """
        Create a parameterised command with validated parameters.
        :param command_type:  The type of command
        :param parameters: command-specific parameters as keyword arguments
        :return: Dictionary with command type and validated parameters
        """

        if command_type not in self.parameter_ranges:
            return {'command': command_type}  # Command does not need parameters

        validated_params = {}
        param_ranges = self.parameter_ranges[command_type]

        # Use provided parameters or defaults, ensuring their within valid ranges
        for param_name, (min_val, max_val) in param_ranges.items():
            if param_name in parameters:
                value = parameters[param_name]
                # Ensure value is within range
                validated_params[param_name] = max(min_val, min(max_val, value))
            else:
                # Use default value
                validated_params[param_name] = self.default_values[command_type].get(param_name)

        return {
            'command': command_type,
            'parameters': validated_params
        }

    def parse_command_text(self, command_text):

        """
        Extract parameters from the natural language command text.
        :param command_text: Natural language command
        :return: command type and extracted parameters
        """

        command_text = command_text.lower()

        # Initialise with no command and empty parameters
        command_type = None
        params = {}

        # Identify the command type
        if any(phrase in command_text for phrase in ['adjust trajectory', 'change course']):
            command_type = 'adjust_trajectory'

            # Look for angle specifications
            if 'degrees x' in command_text or 'x axis' in command_text:
                x_match = re.search(r'(\-?\d+(?:\.\d+)?)\s*(?:degrees?|°)?\s*(?:in|on|along)?\s*(?:the)?\s*x',
                                    command_text)
                if x_match:
                    params['angle_x'] = float(x_match.group(1))

            if 'degrees y' in command_text or 'y axis' in command_text:
                y_match = re.search(r'(\-?\d+(?:\.\d+)?)\s*(?:degrees?|°)?\s*(?:in|on|along)?\s*(?:the)?\s*y',
                                    command_text)
                if y_match:
                    params['angle_y'] = float(y_match.group(1))

            if 'degrees z' in command_text or 'z axis' in command_text:
                z_match = re.search(r'(\-?\d+(?:\.\d+)?)\s*(?:degrees?|°)?\s*(?:in|on|along)?\s*(?:the)?\s*z',
                                    command_text)
                if z_match:
                    params['angle_z'] = float(z_match.group(1))

            # Look for thrust magnitude
            thrust_match = re.search(r'(?:thrust|power|magnitude)\s*(?:of|at)?\s*(\d+(?:\.\d+)?)\s*(?:percent|%)?',
                                     command_text)
            if thrust_match:
                magnitude = float(thrust_match.group(1))
                # Convert percentage to 0-1 scale if needed
                if 'percent' in command_text or '%' in command_text:
                    magnitude /= 100
                params['magnitude'] = magnitude

        elif any(phrase in command_text for phrase in ['increase velocity', 'speed up', 'accelerate']):
            command_type = 'increase_velocity'
            # Look for velocity change
            velocity_match = re.search(r'(?:by|at|to)\s*(\d+(?:\.\d+)?)\s*(?:units?|meters?|m)?(?:/s(?:ec)?)?',
                                       command_text)
            if velocity_match:
                params['delta_v'] = float(velocity_match.group(1))

        elif any(phrase in command_text for phrase in ['decrease velocity', 'slow down', 'decelerate']):
            command_type = 'decrease_velocity'
            # Look for velocity change
            velocity_match = re.search(r'(?:by|at|to)\s*(\d+(?:\.\d+)?)\s*(?:units?|meters?|m)?(?:/s(?:ec)?)?',
                                       command_text)
            if velocity_match:
                params['delta_v'] = float(velocity_match.group(1))

        elif 'investigate' in command_text or 'scan' in command_text or 'examine' in command_text:
            command_type = 'investigate_anomaly'
            # Look for scan intensity
            intensity_match = re.search(r'(?:intensity|level|detail)\s*(?:of|at)?\s*(\d+)', command_text)
            if intensity_match:
                params['scan_intensity'] = int(intensity_match.group(1))

            # Look for approach distance
            distance_match = re.search(r'(?:distance|approach|range)\s*(?:of|at)?\s*(\d+)\s*(?:meters?|m)',
                                       command_text)
            if distance_match:
                params['approach_distance'] = int(distance_match.group(1))

        elif 'refuel' in command_text or 'fuel' in command_text:
            command_type = 'refuel'
            # Look for the amount
            amount_match = re.search(r'(?:amount|level|percent|%)?\s*(?:of|at)?\s*(\d+(?:\.\d+)?)\s*(?:percent|%)?',
                                     command_text)
            if amount_match:
                amount = float(amount_match.group(1))
                # Convert percentage to 0-1 scale if needed
                if 'percent' in command_text or '%' in command_text:
                    amount /= 100
                params['amount'] = amount

        if command_type:
            # Validate and create the command
            return self.create_command(command_type, **params)
        else:
            print("Command not recognised. Please try again.")
            return None



