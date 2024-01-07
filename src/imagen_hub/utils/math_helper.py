def sigfig(number, round=4, digit_mode=True):
    """
    Convert a number to its significant figure representation.
    
    Args:
        number (float/list): Number or list of numbers to convert.
        round (int, optional): Number of significant figures to keep. Defaults to 4.
        digit_mode (bool, optional): If set to True, will use the digit mode for formatting. Defaults to True.
        
    Returns:
        float/list: Number(s) in their significant figure representation.
    """
    if digit_mode:
        string_mode = '{:#.{sigfigs}f}'
    else:
        string_mode = '{:#.{sigfigs}g}'
    if isinstance(number, list):
        new_numbers = []
        for num in number:
            new_num = string_mode.format(num, sigfigs=round)
            new_numbers.append(float(new_num))
        return new_numbers
    else:
        return float(string_mode.format(number, sigfigs=round))
