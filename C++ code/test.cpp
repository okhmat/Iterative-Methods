# include <iostream>
# include <fstream>

using namespace std;

int main(int argc, char *argv[])
{
    ifstream ifs("./Linear_Systems/A.csv", ifstream::in);

    cout << "ifs.good: " << ifs.good() << endl;

    char c = ifs.get();

    int i = 0;
    while(ifs.good() and i<20)
    {
        cout << c;
        c = ifs.get();
        ++i;
    }

    ifs.close();


    return 0;
}

