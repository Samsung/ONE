/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

var http = require('http');
var fs = require('fs');
var os = require('os');
var cp = require('child_process');

var server = {};

server.Server = class {
  constructor() {
    this._ipaddr = undefined;
    this._port = 7070;
    this._run_path = __dirname;
    this._load_file = undefined;
  }

  ipAddrs() {
    let available = [];
    let interfaces = os.networkInterfaces();
    for (let name in interfaces) {
      let ifaces = interfaces[name];
      for (let i = 0; i < ifaces.length; i++) {
        let iface = ifaces[i];
        if (iface.family === 'IPv4' && !iface.internal) {
          available.push(iface.address);
        }
      }
    }
    if (available.length === 0) {
      available.push('127.0.0.1');
    }
    return available;
  }

  help() {
    console.log('Usage: one-visquv [-h] [-i ADDR] [-p PORT] ErrorFile');
    console.log('');
    console.log('optional arguments:');
    console.log('  -h, --help          Show help');
    console.log('  -i, --ip   ADDR     Listen to ADDR address');
    console.log('  -p, --port PORT     Listen to PORT port, default is ' + this._port);
  }

  parseArgs() {
    let args = process.argv;
    for (let idx=2; idx < args.length; ++idx) {
      if (args[idx] === '--help' || args[idx] === '-h') {
        this.help();
        process.exit();
      }
    }
    for (let idx=2; idx < args.length; ++idx) {
      if (args[idx] === '--ip' || args[idx] === '-i') {
        idx++;
        this._ipaddr = args[idx];
      } else if (args[idx] === '--port' || args[idx] === '-p') {
        idx++;
        this._port = args[idx];
      } else if (args[idx] === '--run_path') {
        // this is hidden option to set run_path, where one-visq was called
        idx++;
        this._run_path = args[idx];
      } else if (args[idx].startsWith('-')) {
        console.log('Unknown option: ' + args[idx]);
        process.exit();
      } else {
        this._load_file = args[idx];
      }
    }

    if (this._load_file === undefined) {
      console.log("ErrorFile unknow.")
      process.exit(1);
    }
  }

  start() {
    let ipaddrs = this.ipAddrs();
    ipaddrs.forEach((address) => {
      console.log("Available IP: " + address);
    });

    this._ipaddr = ipaddrs[0];
    this.parseArgs();

    let thiz = this;
    let app = http.createServer(function(request, response) {
      let url = request.url;
      let filepath = __dirname + url;

      if (url === '/') {
        filepath = __dirname + '/visquv-view.html';
      }
      if (url === '/THE_MODEL_FILE') {
        url = thiz._load_file;
        if (thiz._load_file.startsWith('/')) {
          // file is absolute path.
          filepath = thiz._load_file;
        } else {
          filepath = thiz._run_path + '/' + thiz._load_file;
        }
      }
      if (!fs.existsSync(filepath)) {
        console.log("File not found:", filepath);
        return response.writeHead(404);
      }
      console.log("Send:", filepath);
      let data = fs.readFileSync(filepath);
      response.writeHead(200);
      response.end(data);
    });

    console.log('Listening: ' + this._ipaddr + ':' + this._port);
    app.listen(this._port, this._ipaddr, () => {
      let svrinfo = app.address();
      let url = 'http://' + svrinfo.address + ':' + svrinfo.port + '/';
      console.log('Open in browser with ' + url);
      console.log('Use Control-C to stop this visq-server.')
      // enable open browser
      // let cmd = (process.platform === 'linux' ? 'xdg-open' :
      //            process.platform === 'win32' ? 'start' : 'open');
      // cp.exec(`${cmd} ${url}`);
    });
  }
}


var main = new server.Server();
main.start();
