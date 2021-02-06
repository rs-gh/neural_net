import numpy as np

DEBUG_LOGIC = False
DEBUG_CALCULATIONS = False


def debug_logic(*args):
    if DEBUG_LOGIC:
        print(*args)

def debug_calculations(*args):
    if DEBUG_CALCULATIONS:
        print(*args)


def verbose_print(*args, verbose=True):
    if verbose:
        print(*args)


def _zero_pad(inp, final_height, final_width):
    channels, input_height, input_width = inp.shape
    height_diff = final_height-input_height
    width_diff = final_width-input_width
    height_padding_upper = np.zeros((channels, int(height_diff/2), input_width))
    height_padding_lower = np.zeros((channels, int(height_diff/2)+1 if height_diff % 2 else int(height_diff/2), input_width))
    width_padding_left = np.zeros((channels, final_height, int(width_diff/2)))
    width_padding_right = np.zeros((channels, final_height, int(width_diff/2)+1 if width_diff % 2 else int(width_diff/2)))

    debug_logic(height_padding_upper.shape)
    debug_logic(height_padding_lower.shape)
    debug_logic(inp.shape)

    x = np.concatenate((
        height_padding_upper,
        inp,
        height_padding_lower
    ), axis=1)
    x = np.concatenate((
        width_padding_left,
        x,
        width_padding_right
    ), axis=2)
    return x


def zero_pad(inp, kernel_size):
    channels, input_height, input_width = inp.shape
    kernel_height, kernel_width = kernel_size
    final_height = input_height
    final_width = input_width
    if input_height%kernel_height:
        final_height = input_height + kernel_height - (input_height%kernel_height)
    if input_width%kernel_width:
        final_width = input_width + kernel_width - (input_width%kernel_width)
    return _zero_pad(inp, final_height, final_width)


def get_conv_input_reshaped(inp):
    debug_logic("get_input_reshaped: current inp shape is: {}".format(inp.shape))
    try:
        num_inputs, input_channels, input_height, input_width = inp.shape
        debug_logic("get_input_reshaped: no error, inp shape is: {}".format(inp.shape))
    except Exception as err:
        inp = inp.reshape((inp.shape[0], 1, inp.shape[1], inp.shape[2]))
        debug_logic("get_input_reshaped: {}, reshaping inp shape to: {}".format(err, inp.shape))
    finally:
        debug_logic("reshaping inp shape to: {}".format(inp.shape))
        return inp


def get_conv_output_reshaped(output):
    try:
        debug_logic("get_conv_output_reshaped: current output shape is: {}".format(output.shape))
        num_inputs, num_outputs, num_output_channels, output_height, output_width = output.shape
        if num_outputs == 1:
            output = output.reshape((num_inputs, num_output_channels, output_height, output_width))
    except Exception as err:
        debug_logic(err)
    finally:
        debug_logic("get_conv_output_reshaped: reshaped output shape is: {}".format(output.shape))
        return output


def get_dense_input_reshaped(inp):
    debug_logic("get_dense_input_reshaped: current inp shape is: {}".format(inp.shape))
    try:
        num_inputs, input_dim = inp.shape
        debug_logic("get_dense_input_reshaped: no error")
    except Exception as err:
        inp = inp.reshape((1, *inp.shape))
        debug_logic("get_dense_input_reshaped: {}, reshaping inp shape to: {}".format(err, inp.shape))
    finally:
        debug_logic("reshaping inp shape to: {}".format(inp.shape))
        return inp


def get_kernel_reshaped(kernel):
    try:
        kernel_inp_channels, output_channels, kernel_height, kernel_width = kernel.shape
    except ValueError:
        kernel = kernel.reshape((3, 1, *kernel.shape))
    finally:
        return kernel


def generate_batch(inp, labels, num_samples, batch_size):
    for i in range(0, num_samples, batch_size):
        yield i, inp[i:i+batch_size, :, :], labels[i:i+batch_size, :]
