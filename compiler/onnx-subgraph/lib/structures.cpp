#include "partition.h"
#include <algorithm>
int DetermineStructure(const onnx::GraphProto& graph, Device &d,PartitionStrategy strategy) 
{
    int node_index = 0;
    std::vector<std::vector<std::string>> enabled_structure;
    std::vector<std::string> structure_temp;
    while(node_index < graph.node_size())
    {
        std::vector<std::string> support_op;
        const auto& node = graph.node(node_index);
        switch (strategy)
        {
            case SPILTE_CPU_STRUCTURE_FIRST:
            {
                support_op =d.getCPUSupportOp();
                break;
            }
            case SPILTE_NPU_STRUCTURE_FIRST:
            {
                support_op =d.getNPUSupportOp();
                break;
            }
            default: 
            {break;}
        }
    if(std::find(support_op.begin(),support_op.end(),node.op_type())!=support_op.end())
            {
                auto op_index=std::find(support_op.begin(),support_op.end(),node.op_type());
                structure_temp.push_back(*op_index);
            }
            else
            {
                if(structure_temp.size()>=3)
                {
                    bool isequal=0;
                    for(const auto& structure : enabled_structure)
                    
                    {
                        if(std::equal(structure.begin(),structure.end(),structure_temp.begin(),structure_temp.end()))
                        {
                            isequal=1;
                            break;
                        }
                    }
                    if(isequal==0)
                    {
                        enabled_structure.push_back(structure_temp);
                    }
                }
                if(structure_temp.size()!=0){
                    structure_temp.clear();
                }
                
            }
        node_index++;
    }
    
        for(const auto& structure : enabled_structure)
        {
            std::cout<<"{";
            for(const auto& op : structure)
            {
                std::cout <<"\""<< op << "\",";
            }
            std::cout<<"},"<<std::endl;
        }
    return 0;
}
