import os 
import csv
import argparse

def generate_sequence(beam_dir, writer):
    beam_names = os.listdir(beam_dir)
    cam_names = [[], [], []]
    num_data = len(beam_names)

    # Split data based on camera ID
    for i in range(num_data):
        split_name = beam_names[i].split('_')
        if split_name[1] == '1':
            cam_names[0].append(beam_names[i])
        elif split_name[1] == '2':
            cam_names[1].append(beam_names[i])
        elif split_name[1] == '3':
            cam_names[2].append(beam_names[i])

    
    for i in range(3):
        num_cam_data = len(cam_names[i])
        trace_names = [[], [], [], [], []]

        for j in range(num_cam_data):
            split_name = cam_names[i][j].split('_')
            if split_name[3] == '9':
                trace_names[0].append(cam_names[i][j])
            elif split_name[3] == '9.5':
                trace_names[1].append(cam_names[i][j])
            elif split_name[3] == '10':
                trace_names[2].append(cam_names[i][j])
            elif split_name[3] == '10.5':
                trace_names[3].append(cam_names[i][j])
            elif split_name[3] == '11':
                trace_names[4].append(cam_names[i][j])

        for j in range(5):
            num_trace_data = len(trace_names[j])
            # Sort the data based on x axis
            for k in range(num_trace_data):
                for l in range(num_trace_data-k-1):
                    x_axis_current = float(trace_names[j][l].split('_')[2])
                    x_axis_next = float(trace_names[j][l+1].split('_')[2])
                    if x_axis_current > x_axis_next:
                        trace_names[j][l], trace_names[j][l+1] = trace_names[j][l+1], trace_names[j][l]

            for k in range(num_trace_data-12):
                sequence = []
                for l in range(13):
                    sequence.append(trace_names[j][k+l].split('_')[5].split('.')[0])
                for l in range(13):
                    sequence.append('./rgb/' + trace_names[j][k+l].replace('_'+trace_names[j][k+l].split('_')[-1], '.jpg'))
                for l in range(13):
                    sequence.append('./beam/' + trace_names[j][k+l])
                writer.writerow(sequence)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Generate Beam/Image Sequence Dataset."
    )
    parser.add_argument(
        "--beam_dir", required=True, type=str,
        help="Directory of beam files"
    )
    args = parser.parse_args()

    with open(args.beam_dir.replace('beam', '')+'data.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        header = []
        for i in range(13):
            header.append('Beam {}'.format(i+1))
        for i in range(13):
            header.append('Img Path {}'.format(i+1))
        for i in range(13):
            header.append('Data Rate {}'.format(i+1))
        writer.writerow(header)
        generate_sequence(args.beam_dir, writer)