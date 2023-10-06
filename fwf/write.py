c = 299792458  # m.s-1


def r_to_t(r, n=1.):
    return r / (10**-12 * c) * n * 2


def add_point(record, point_idx, return_number, r_peak, i_peak, las_i, idx_wfrm):

    # resize the ScaleAwarePointRecord
    if point_idx == len(record):
        record.resize(len(record) + 10000)
        print(f"[add_point] increase the size of the record => {len(record)}")

    # copy the fields from the anchor point record
    record.array[point_idx] = las_i.points[idx_wfrm].array
    peak_return_point_wave_location = r_to_t(r_peak)

    return_point_wave_location = record.return_point_wave_location[point_idx]
    X = record.X[point_idx]
    Y = record.Y[point_idx]
    Z = record.Z[point_idx]
    x_t = record.x_t[point_idx]
    y_t = record.y_t[point_idx]
    z_t = record.z_t[point_idx]

    anchor_x = X + x_t * return_point_wave_location * 1000
    anchor_y = Y + y_t * return_point_wave_location * 1000
    anchor_z = Z + z_t * return_point_wave_location * 1000
    new_x = anchor_x - x_t * peak_return_point_wave_location * 1000
    new_y = anchor_y - y_t * peak_return_point_wave_location * 1000
    new_z = anchor_z - z_t * peak_return_point_wave_location * 1000

    # update the las fields coordinates
    record.X[point_idx] = new_x
    record.Y[point_idx] = new_y
    record.Z[point_idx] = new_z

    # return_point_wave_location
    record.return_point_wave_location[point_idx] = peak_return_point_wave_location

    # intensity
    record.intensity[point_idx] = i_peak

    record.return_number[point_idx] = return_number
    if return_number == 2:
        record.number_of_returns[point_idx - 1] = 2
        record.number_of_returns[point_idx] = 2
    else:
        record.number_of_returns[point_idx] = 1

    return point_idx + 1


def add_point_copy(record, point_idx, las_data, idx_wfrm):

    # resize the ScaleAwarePointRecord
    if point_idx == len(record):
        record.resize(len(record) + 10000)
        print(f"[add_point] increase the size of the record => {len(record)}")

    # copy the fields from the anchor point record
    record.array[point_idx] = las_data.points.array[idx_wfrm]
