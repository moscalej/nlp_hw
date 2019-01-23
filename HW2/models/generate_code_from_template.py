args_dict = {}
args_dict['d(hd)'] = {'valid': '#TRUE#', 'value': '#d_h_d#'}
args_dict['wh'] = {'valid': '#TRUE#', 'value': '#w_h#'}
args_dict['ph'] = {'valid': '#TRUE#', 'value': '#p_h#'}
args_dict['wh-1'] = {'valid': '#valid_prev_h#'}
args_dict['ph-1'] = {'valid': '#valid_prev_h#'}
args_dict['ph-1']['value'] = '#p_h_prev#'
args_dict['wh-1']['value'] = '#w_h_prev#'
args_dict['wh+1'] = {'valid': '#valid_next_h#'}
args_dict['ph+1'] = {'valid': '#valid_next_h#'}
args_dict['wh+1']['value'] = '#w_h_next#'
args_dict['ph+1']['value'] = '#p_h_next#'
args_dict['wd'] = {'valid': '#TRUE#', 'value': '#w_d#'}
args_dict['pd'] = {'valid': '#TRUE#', 'value': '#p_d#'}
args_dict['wd-1'] = {'valid': '#valid_prev_d#'}
args_dict['pd-1'] = {'valid': '#valid_prev_d#'}
args_dict['pd-1']['value'] = '#p_d_prev#'
args_dict['wd-1']['value'] = '#w_d_prev#'
args_dict['wd+1'] = {'valid': '#valid_next_d#'}
args_dict['pd+1'] = {'valid': '#valid_next_d#'}
args_dict['wd+1']['value'] = '#w_d_next#'
args_dict['pd+1']['value'] = '#p_d_next#'
args_dict['wb'] = {'valid': '#valid_b#'}
args_dict['pb'] = {'valid': '#valid_b#'}
args_dict['wb-1'] = {'valid': '#valid_prev_b#'}
args_dict['pb-1'] = {'valid': '#valid_prev_b#'}
args_dict['wb+1'] = {'valid': '#valid_next_b#'}
args_dict['pb+1'] = {'valid': '#valid_next_b#'}
args_dict['wb'] = {'valid': '#TRUE#', 'value': '#w_b#'}
args_dict['pb'] = {'valid': '#TRUE#', 'value': '#p_b#'}
args_dict['wb-1'] = {'valid': '#valid_prev_b#'}
args_dict['pb-1'] = {'valid': '#valid_prev_b#'}
args_dict['pb-1']['value'] = '#p_b_prev#'
args_dict['wb-1']['value'] = '#w_b_prev#'
args_dict['wb+1'] = {'valid': '#valid_next_b#'}
args_dict['pb+1'] = {'valid': '#valid_next_b#'}
args_dict['wb+1']['value'] = '#w_b_next#'
args_dict['pb+1']['value'] = '#p_b_next#'
args_dict['wc'] = {'valid': '#valid_c#'}
args_dict['pc'] = {'valid': '#valid_c#'}
args_dict['wc-1'] = {'valid': '#valid_c#'}
args_dict['pc-1'] = {'valid': '#valid_c#'}
args_dict['wc+1'] = {'valid': '#valid_c#'}
args_dict['pc+1'] = {'valid': '#valid_c#'}
args_dict['d(hdc)'] = {'valid': '#valid_c#'}
args_dict['d(hdc)'] = {'valid': '#valid_c#', 'value': '#d_h_d + d_d_c#'}
args_dict['wc'] = {'valid': '#valid_c#', 'value': '#w_c#'}
args_dict['pc'] = {'valid': '#valid_c#', 'value': '#p_c#'}
args_dict['wc-1'] = {'valid': '#valid_prev_h#'}
args_dict['pc-1'] = {'valid': '#valid_prev_h#'}
args_dict['pc-1']['value'] = '#p_c_prev#'
args_dict['wc-1']['value'] = '#w_c_prev#'
args_dict['wc+1'] = {'valid': '#valid_next_h#'}
args_dict['pc+1'] = {'valid': '#valid_next_h#'}
args_dict['wc+1']['value'] = '#w_c_next#'
args_dict['pc+1']['value'] = '#p_c_next#'
keys = []


def get_key(name, *args):
    return ' '.join((name,) + tuple(args))


def add_from_temp(keys, sig, args_dict, acc_args=[], acc_valid='#TRUE', acc_name=''):
    # example sig: '[wp]h-1, [wp]h, pc-1, [wp]c, d(h, d, c)'
    # args_dict: args_dict[w_h]-> {[valid]->True [value]->5}
    if not acc_valid:
        return
    parts = [part.replace(' ', '') for part in sig.split(',')]  # split and remove spaces
    num_parts = len(parts)
    if num_parts == 0 or len(parts[0]) == 0:
        # self._add_key(keys, True, acc_name, *acc_args)
        keys.append(get_key(acc_valid + '%' + acc_name + '%', *acc_args))
        return
    current = parts[0]
    if current[0] == '[':
        next_typ = current[4:]
        p_part = 'p' + next_typ + ',' + ','.join(parts[1:])
        w_part = 'w' + next_typ + ',' + ','.join(parts[1:])
        add_from_temp(keys, p_part, args_dict, acc_args, acc_valid, acc_name)
        add_from_temp(keys, w_part, args_dict, acc_args, acc_valid, acc_name)
    else:
        acc_valid_ = acc_valid + '#and' + args_dict[current]['valid']
        if not acc_valid_:
            return
        acc_args_ = acc_args + [',' + args_dict[current]['value']]
        acc_name_ = acc_name + current
        add_from_temp(keys, ','.join(parts[1:]), args_dict, acc_args_, acc_valid_, acc_name=acc_name_)


add_from_temp(keys, f'[wp]h,[wp]d,d(hd)', args_dict)
add_from_temp(keys, f'[wp]h,d(hd)', args_dict)
add_from_temp(keys, f'wd, pd, d(hd)', args_dict)
add_from_temp(keys, f'[wp]d, d(hd)', args_dict)
add_from_temp(keys, f'wh, ph, wd, pd, d(hd)', args_dict)
add_from_temp(keys, f'ph, wh, pd, d(hd)', args_dict)
add_from_temp(keys, f'wh, wd, pd, d(hd)', args_dict)
add_from_temp(keys, f'wh, ph, [wp]d, d(hd)', args_dict)
add_from_temp(keys, f'wh, ph, [wp]d, d(hd)', args_dict)
add_from_temp(keys, f'ph, pb, pd, d(hd)', args_dict)
add_from_temp(keys, f'ph, ph+1, pd-1, pd, d(hd)', args_dict)
add_from_temp(keys, f'ph-1, ph, pd-1, pd, d(hd)', args_dict)
add_from_temp(keys, f'ph, ph+1, pd, pd+1, d(hd)', args_dict)
add_from_temp(keys, f'ph-1, ph, pd, pd+1, d(hd)', args_dict)
add_from_temp(keys, f'[wp]h, [wp]c, [wp]c+1, d(hdc)', args_dict)
add_from_temp(keys, f'ph, pd, pc, d(hdc)', args_dict)
add_from_temp(keys, f'wh, wd, wc, d(hdc)', args_dict)
add_from_temp(keys, f'ph, [wp]c, d(hdc)', args_dict)
add_from_temp(keys, f'pd, [wp]c, d(hdc)', args_dict)
add_from_temp(keys, f'wh, [wp]c, d(hdc)', args_dict)
add_from_temp(keys, f'wd, [wp]c, d(hdc)', args_dict)
add_from_temp(keys, f'[wp]h, [wp]h+1, [wp]c, d(hdc)', args_dict)
add_from_temp(keys, f'[wp]h-1, [wp]h, [wp]c, d(hdc)', args_dict)
add_from_temp(keys, f'[wp]h, [wp]c-1, [wp]c, d(hdc)', args_dict)
add_from_temp(keys, f'[wp]h, [wp]c, [wp]c+1, d(hdc)', args_dict)
add_from_temp(keys, f'[wp]h-1, [wp]h, [wp]c-1, [wp]c, d(hdc)', args_dict)
add_from_temp(keys, f'[wp]h-1, [wp]h, [wp]c-1, [wp]c, d(hdc)', args_dict)
add_from_temp(keys, f'[wp]h-1, [wp]h, [wp]c-1, [wp]c, d(hdc)', args_dict)
add_from_temp(keys, f'[wp]h, [wp]h+1, [wp]c-1, [wp]c, d(hdc)', args_dict)
add_from_temp(keys, f'[wp]h-1, [wp]h, [wp]c, [wp]c+1, d(hdc)', args_dict)
add_from_temp(keys, f'[wp]h, [wp]h+1, [wp]c, [wp]c+1, d(hdc)', args_dict)
add_from_temp(keys, f'[wp]d, [wp]d+1, [wp]c, d(hdc)', args_dict)
add_from_temp(keys, f'[wp]d-1, [wp]d, [wp]c, d(hdc)', args_dict)
add_from_temp(keys, f'[wp]d, [wp]c-1, [wp]c, d(hdc)', args_dict)
add_from_temp(keys, f'[wp]d, [wp]c, [wp]c+1, d(hdc)', args_dict)
add_from_temp(keys, f'[wp]d, [wp]d+1, [wp]c-1, [wp]c, d(hdc)', args_dict)
add_from_temp(keys, f'[wp]d, [wp]d+1, [wp]c, [wp]c+1, d(hdc)', args_dict)
add_from_temp(keys, f'[wp]d-1, [wp]d, [wp]c-1, [wp]c, d(hdc)', args_dict)
add_from_temp(keys, f'[wp]d-1, [wp]d, [wp]c, [wp]c+1, d(hdc)', args_dict)
filepath = r'C:\Users\afinkels\Desktop\private\Technion\Master studies\עיבוד שפה טבעית\HW\hw_repo\nlp_hw\HW2\models\keys.txt'
file_handler = open(filepath, 'w')
for item in keys:
    file_handler.write("{}\n".format(item))

file_handler.close()
