
```markdown
To install Hadoop and set up HDFS locally on your Windows machine using WSL, the following steps were successfully executed after troubleshooting. These steps assume a clean setup in WSL’s Linux environment (Ubuntu) with the user `patwh`. Commands that resulted in errors are excluded, and troubleshooting tips for major steps are provided.

## Summary of Successful Steps to Install Hadoop

1. **Install WSL and Set Up Environment**:
   - **Action**: Installed WSL and started a Ubuntu session.
     ```cmd
     wsl --install
     wsl -d Ubuntu
     ```
   - **Purpose**: Provided a Linux environment for Hadoop, which is Linux-native.
   - **Troubleshooting**:
     - Verify WSL installation: `wsl --list --verbose` (should show Ubuntu as `Running`).
     - If commands fail, reset WSL: `wsl --unregister Ubuntu` and re-run `wsl --install`.

2. **Install Java**:
   - **Action**: Installed OpenJDK 8 in WSL.
     ```bash
     sudo apt update
     sudo apt install openjdk-8-jdk
     ```
   - **Purpose**: Hadoop requires Java (JDK 8 recommended).
   - **Troubleshooting**:
     - Verify: `java -version` (should show OpenJDK 8).
     - If missing, ensure `apt` is updated or check for network issues.

3. **Download and Extract Hadoop**:
   - **Action**: Downloaded Hadoop 3.3.6 and extracted it in `/home/patwh`.
     ```bash
     cd /home/patwh
     wget https://downloads.apache.org/hadoop/common/hadoop-3.3.6/hadoop-3.3.6.tar.gz
     tar -xvzf hadoop-3.3.6.tar.gz
     ```
   - **Purpose**: Installed Hadoop binaries in WSL’s Linux file system.
   - **Troubleshooting**:
     - Verify download: `ls -lh | grep hadoop` (should show `hadoop-3.3.6.tar.gz`).
     - Check extraction: `ls -lh hadoop-3.3.6` (should show `bin`, `etc`, `lib`).
     - If slow or fails, check disk space: `df -h`. Avoid `/mnt/c` to prevent performance issues.

4. **Set Environment Variables**:
   - **Action**: Added `JAVA_HOME` and `HADOOP_HOME` to `~/.bashrc`.
     ```bash
     nano ~/.bashrc
     ```
     Added:
     ```bash
     export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64
     export HADOOP_HOME=/home/patwh/hadoop-3.3.6
     export PATH=$PATH:$HADOOP_HOME/bin
     ```
     Applied:
     ```bash
     source ~/.bashrc
     ```
   - **Purpose**: Enabled Hadoop to locate Java and its binaries.
   - **Troubleshooting**:
     - Verify: `echo $JAVA_HOME` and `echo $HADOOP_HOME` (should show correct paths).
     - If empty, re-edit `~/.bashrc` and check for typos. Ensure JDK path is correct: `readlink -f $(which java)`.

5. **Configure Hadoop**:
   - **Action**: Edited configuration files in `/home/patwh/hadoop-3.3.6/etc/hadoop`.
     - `core-site.xml`:
       ```bash
       nano /home/patwh/hadoop-3.3.6/etc/hadoop/core-site.xml
       ```
       Added:
       ```xml
       <configuration>
           <property>
               <name>fs.defaultFS</name>
               <value>hdfs://localhost:9000</value>
           </property>
       </configuration>
       ```
     - `hdfs-site.xml`:
       ```bash
       nano /home/patwh/hadoop-3.3.6/etc/hadoop/hdfs-site.xml
       ```
       Added:
       ```xml
       <configuration>
           <property>
               <name>dfs.replication</name>
               <value>1</value>
           </property>
           <property>
               <name>dfs.namenode.name.dir</name>
               <value>file:///home/patwh/hadoop-3.3.6/data/namenode</value>
           </property>
           <property>
               <name>dfs.datanode.data.dir</name>
               <value>file:///home/patwh/hadoop-3.3.6/data/datanode</value>
           </property>
       </configuration>
       ```
     - `hadoop-env.sh`:
       ```bash
       nano /home/patwh/hadoop-3.3.6/etc/hadoop/hadoop-env.sh
       ```
       Added:
       ```bash
       export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64
       export HDFS_NAMENODE_USER=patwh
       export HDFS_DATANODE_USER=patwh
       export HDFS_SECONDARYNAMENODE_USER=patwh
       export HADOOP_DATANODE_OPTS="-Dcom.sun.management.jmxremote $HADOOP_DATANODE_OPTS"
       ```
   - **Purpose**: Configured HDFS for single-node operation and set user permissions.
   - **Troubleshooting**:
     - Verify files: `cat /home/patwh/hadoop-3.3.6/etc/hadoop/*.xml` and `cat /home/patwh/hadoop-3.3.6/etc/hadoop/hadoop-env.sh`.
     - If errors, check XML syntax or re-copy configurations.

6. **Set Up SSH**:
   - **Action**: Configured passwordless SSH for Hadoop.
     ```bash
     sudo apt install openssh-server
     sudo service ssh start
     ssh-keygen -t rsa -P '' -f ~/.ssh/id_rsa
     cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys
     chmod 600 ~/.ssh/authorized_keys
     ssh localhost
     exit
     ```
   - **Purpose**: Enabled Hadoop to manage processes via SSH.
   - **Troubleshooting**:
     - Verify SSH: `sudo service ssh status` (should show `active (running)`).
     - If `Connection refused`, restart SSH: `sudo service ssh restart`. Check port 22: `sudo netstat -tuln | grep 22`.

7. **Create Data Directories**:
   - **Action**: Created and set permissions for Namenode and Datanode directories.
     ```bash
     mkdir -p /home/patwh/hadoop-3.3.6/data/namenode
     mkdir -p /home/patwh/hadoop-3.3.6/data/datanode
     sudo chown -R patwh:patwh /home/patwh/hadoop-3.3.6/data
     sudo chmod -R 755 /home/patwh/hadoop-3.3.6/data
     ```
   - **Purpose**: Provided storage for HDFS metadata and data.
   - **Troubleshooting**:
     - Verify: `ls -ld /home/patwh/hadoop-3.3.6/data/{namenode,datanode}` (should show `drwxr-xr-x ... patwh patwh`).
     - If missing, recreate directories. If permission errors, re-run `chown`/`chmod`.

8. **Format Namenode**:
   - **Action**: Formatted the Namenode to initialize HDFS.
     ```bash
     hdfs namenode -format
     ```
     Typed `Y` at the prompt: `Re-format filesystem ... ? (Y or N)`.
   - **Purpose**: Initialized the HDFS filesystem.
   - **Troubleshooting**:
     - Confirm completion: Look for `SHUTDOWN_MSG`.
     - If errors, check logs: `cat /home/patwh/hadoop-3.3.6/logs/hadoop-*.log`.

9. **Start HDFS**:
   - **Action**: Started HDFS processes.
     ```bash
     /home/patwh/hadoop-3.3.6/sbin/start-dfs.sh
     ```
   - **Purpose**: Launched Namenode, Datanode, and Secondary Namenode.
   - **Troubleshooting**:
     - Verify: `jps` (should list `NameNode`, `DataNode`, `SecondaryNameNode`).
     - Check UI: `http://localhost:9870`.
     - If fails, check logs: `cat /home/patwh/hadoop-3.3.6/logs/hadoop-*.log`. Ensure ports 9000, 9870, 9866 are free: `sudo netstat -tuln | grep ':9000\|:9870\|:9866'`.

10. **Verify HDFS**:
    - **Action**: Tested HDFS functionality.
      ```bash
      hdfs dfs -mkdir /test
      hdfs dfs -ls /
      ```
    - **Purpose**: Confirmed HDFS is operational.
    - **Troubleshooting**:
      - If commands fail, check `jps` and logs. Ensure Namenode is running (`http://localhost:9870`).
      - Verify user permissions in `hadoop-env.sh`.

## Additional Notes
- **Environment**: All steps were performed in WSL’s `/home/patwh` to avoid Windows file system (`/mnt/c`) performance issues.
- **User**: Used `patwh` as the Linux user, avoiding earlier `mypc\patwh` complications.
- **Ignored Errors**: Commands like `sudo chown -R mypc\\patwh:mypc\\patwh` (failed due to invalid user) and earlier `start-dfs.sh` errors (e.g., SSH, permissions, missing directories) were resolved by the above steps.
- **Next Steps**: You can now use HDFS for data storage (e.g., `hdfs dfs -put file.txt /test`) or configure YARN for MapReduce jobs (as outlined previously).

Your HDFS setup is complete! If you encounter issues, check logs (`/home/patwh/hadoop-3.3.6/logs`) and verify `jps` output.
```