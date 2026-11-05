import sys
import os # Import os module for path joining

def process_file(input_path):
    """
    Reads an input text file, removes empty lines and surrounding whitespace,
    joins the remaining lines into a single line separated by newline characters,
    and writes the result to 'output.txt' in the current directory.

    Args:
        input_path (str): Path to the input text file. This will be hardcoded in the main block.
    """
    output_path = "output.txt" # Hardcoded output path
    try:
        with open(input_path, 'r', encoding='utf-8') as infile:
            lines = infile.readlines()
    except FileNotFoundError:
        print(f"错误: 输入文件未找到 '{input_path}'", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"读取文件时出错 '{input_path}': {e}", file=sys.stderr)
        sys.exit(1)

    # 处理行: 移除空行和首尾空格
    processed_lines = [line.strip() for line in lines if line.strip()]

    # 用换行符连接处理后的行
    single_line_output = r'\n'.join(processed_lines) # 使用 \n 作为分隔符连接行

    try:
        with open(output_path, 'w', encoding='utf-8') as outfile:
            outfile.write(single_line_output)
        print(f"处理完成. 结果已写入 '{output_path}'")
    except Exception as e:
        print(f"写入文件时出错 '{output_path}': {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    # Define the hardcoded input file path relative to the script's directory parent
    # Assuming the script is in speedtest and the file is in tool
    # script_dir = os.path.dirname(os.path.abspath(__file__))
    # parent_dir = os.path.dirname(script_dir)
    # input_file_path = os.path.join(parent_dir, 'vtsdata.txt') # Use os.path.join for compatibility

    # Alternatively, use the absolute path directly (less portable):
    input_file_path = r'C:\Users\86188\Desktop\workshop\LLMRAG\speedtest\knowledge_base\Tian.txt'

    print(f"正在处理文件: {input_file_path}")
    process_file(input_file_path)
