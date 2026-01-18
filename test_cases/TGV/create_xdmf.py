import h5py
import os

# Configuration
h5_filename = 'NS9_w.h5'
xdmf_filename = 'NS9_w.xdmf'

def generate_xdmf():
    if not os.path.exists(h5_filename):
        print(f"Error: {h5_filename} not found in current directory.")
        return

    with h5py.File(h5_filename, 'r') as f:
        variables = []
        for key in f.keys():
            if isinstance(f[key], h5py.Group) and '3D' in f[key]:
                variables.append(key)
        
        if not variables:
            print("No variables found.")
            return
        
        print(f"Found variables: {variables}")

        first_var = variables[0]
        timesteps = []
        if '3D' in f[first_var]:
            for key in f[first_var]['3D'].keys():
                if key.isdigit():
                    timesteps.append(int(key))
        timesteps.sort()
        print(f"Found {len(timesteps)} timesteps: {timesteps}")
        
        x0 = f[first_var]['mesh']['x0']
        x1 = f[first_var]['mesh']['x1']
        x2 = f[first_var]['mesh']['x2']
        
        n0 = x0.shape[0]
        n1 = x1.shape[0]
        n2 = x2.shape[0]
        
        print(f"Mesh dimensions: {n0}x{n1}x{n2}")

        with open(xdmf_filename, 'w') as xdmf:
            xdmf.write('<?xml version="1.0" ?>\n')
            xdmf.write('<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>\n')
            xdmf.write('<Xdmf Version="2.0">\n')
            xdmf.write(' <Domain>\n')
            xdmf.write('  <Grid Name="TimeSeries" GridType="Collection" CollectionType="Temporal">\n')
            
            for tstep in timesteps:
                # Try to get physical time from attribute
                time_val = tstep
                try:
                    ds_path = f"{first_var}/3D/{tstep}"
                    if ds_path in f:
                        ds = f[ds_path]
                        if 'time' in ds.attrs:
                            time_val = ds.attrs['time']
                except Exception:
                    pass

                xdmf.write(f'   <Grid Name="Mesh" GridType="Uniform">\n')
                xdmf.write(f'    <Time Value="{time_val}"/>\n')
                xdmf.write(f'    <Topology TopologyType="3DRectMesh" Dimensions="{n0} {n1} {n2}"/>\n')
                
                xdmf.write('    <Geometry GeometryType="VXVYVZ">\n')
                xdmf.write(f'     <DataItem Dimensions="{n2}" NumberType="Float" Precision="8" Format="HDF">{h5_filename}:/{first_var}/mesh/x2</DataItem>\n')
                xdmf.write(f'     <DataItem Dimensions="{n1}" NumberType="Float" Precision="8" Format="HDF">{h5_filename}:/{first_var}/mesh/x1</DataItem>\n')
                xdmf.write(f'     <DataItem Dimensions="{n0}" NumberType="Float" Precision="8" Format="HDF">{h5_filename}:/{first_var}/mesh/x0</DataItem>\n')
                xdmf.write('    </Geometry>\n')
                
                for var in variables:
                    ds_path = f"{var}/3D/{tstep}"
                    if ds_path not in f: continue
                    
                    ds = f[ds_path]
                    shape = ds.shape
                    
                    if len(shape) == 3:
                        xdmf.write(f'    <Attribute Name="{var}" AttributeType="Scalar" Center="Node">\n')
                        xdmf.write(f'     <DataItem Dimensions="{n0} {n1} {n2}" NumberType="Float" Precision="8" Format="HDF">{h5_filename}:/{ds_path}</DataItem>\n')
                        xdmf.write('    </Attribute>\n')
                    
                    elif len(shape) == 4 and shape[0] == 3:
                        # 1. Write Vector Attribute (Combined)
                        xdmf.write(f'    <Attribute Name="{var}" AttributeType="Vector" Center="Node">\n')
                        xdmf.write(f'     <DataItem ItemType="Function" Function="JOIN($0, $1, $2)" Dimensions="{n0} {n1} {n2} 3">\n')
                        for i in range(3):
                             xdmf.write(f'      <DataItem ItemType="HyperSlab" Dimensions="{n0} {n1} {n2}" Type="HyperSlab">\n')
                             xdmf.write(f'       <DataItem Dimensions="3 4" Format="XML">{i} 0 0 0 1 1 1 1 1 {n0} {n1} {n2}</DataItem>\n')
                             xdmf.write(f'       <DataItem Dimensions="3 {n0} {n1} {n2}" NumberType="Float" Precision="8" Format="HDF">{h5_filename}:/{ds_path}</DataItem>\n')
                             xdmf.write('      </DataItem>\n')
                        xdmf.write('     </DataItem>\n')
                        xdmf.write('    </Attribute>\n')

                        # 2. Write Scalar Components (Optional but helpful)
                        comps = ['X', 'Y', 'Z']
                        for i, c in enumerate(comps):
                            xdmf.write(f'    <Attribute Name="{var}_{c}" AttributeType="Scalar" Center="Node">\n')
                            xdmf.write(f'     <DataItem ItemType="HyperSlab" Dimensions="{n0} {n1} {n2}" Type="HyperSlab">\n')
                            xdmf.write(f'      <DataItem Dimensions="3 4" Format="XML">{i} 0 0 0 1 1 1 1 1 {n0} {n1} {n2}</DataItem>\n')
                            xdmf.write(f'      <DataItem Dimensions="3 {n0} {n1} {n2}" NumberType="Float" Precision="8" Format="HDF">{h5_filename}:/{ds_path}</DataItem>\n')
                            xdmf.write('     </DataItem>\n')
                            xdmf.write('    </Attribute>\n')

                xdmf.write('   </Grid>\n')
            
            xdmf.write('  </Grid>\n')
            xdmf.write(' </Domain>\n')
            xdmf.write('</Xdmf>\n')
    
    print(f"Successfully generated {xdmf_filename} with Vectors!")

if __name__ == "__main__":
    generate_xdmf()